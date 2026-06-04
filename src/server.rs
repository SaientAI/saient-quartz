use std::convert::Infallible;
use std::sync::Arc;

use axum::{extract::State, response::sse::{Event, KeepAlive, Sse}, routing::{get, post}, Json, Router};
use serde::Deserialize;
use tokio::sync::mpsc;
use tokio_stream::wrappers::ReceiverStream;
use tokio_stream::StreamExt as _;

use crate::{Config, GpuState, Tokenizer, Weights};

pub struct AppState {
    pub cfg:       Arc<Config>,
    pub weights:   Arc<Weights>,
    pub gpu:       Arc<GpuState>,
    pub tokenizer: Arc<Option<Tokenizer>>,
    pub model_id:  String,
}

#[derive(Deserialize)]
struct ChatRequest {
    messages:        Vec<ChatMessage>,
    max_tokens:      Option<usize>,
    temperature:     Option<f32>,
    top_k:           Option<usize>,
    repeat_penalty:  Option<f32>,
    seed:            Option<u64>,
}

#[derive(Deserialize)]
struct ChatMessage {
    role:    String,
    content: String,
}

pub async fn run(port: u16, state: Arc<AppState>) {
    let app = Router::new()
        .route("/health", get(health))
        .route("/v1/models", get(models))
        .route("/v1/chat/completions", post(chat_completions))
        .with_state(state);

    let addr = format!("0.0.0.0:{}", port);
    let listener = tokio::net::TcpListener::bind(&addr).await
        .unwrap_or_else(|e| panic!("cannot bind {}: {}", addr, e));
    eprintln!("tinyq4 server listening on http://{}", addr);
    axum::serve(listener, app).await.unwrap();
}

async fn health() -> &'static str { "ok" }

async fn models(State(state): State<Arc<AppState>>) -> Json<serde_json::Value> {
    Json(serde_json::json!({
        "object": "list",
        "data": [{
            "id": state.model_id.clone(),
            "object": "model",
            "owned_by": "tinyq4"
        }]
    }))
}

async fn chat_completions(
    State(state): State<Arc<AppState>>,
    Json(req): Json<ChatRequest>,
) -> Sse<impl futures_util::Stream<Item = Result<Event, Infallible>>> {
    let max_new        = req.max_tokens.unwrap_or(512);
    let temperature    = req.temperature.unwrap_or(0.0).max(0.0);
    let top_k          = req.top_k.unwrap_or(40).max(1);
    let repeat_penalty = req.repeat_penalty.unwrap_or(1.0).max(1.0);

    // Format messages into prompt token IDs
    let prompt_ids: Vec<usize> = match state.tokenizer.as_ref() {
        Some(tok) => {
            let pairs: Vec<(&str, &str)> = req.messages.iter()
                .map(|m| (m.role.as_str(), m.content.as_str()))
                .collect();
            tok.encode_messages(&pairs).into_iter().map(|id| id as usize).collect()
        }
        None => vec![1],
    };

    let cfg       = Arc::clone(&state.cfg);
    let weights   = Arc::clone(&state.weights);
    let gpu       = Arc::clone(&state.gpu);
    let tokenizer = Arc::clone(&state.tokenizer);

    enum GenEvent {
        Token(String, bool),
        Prefill(usize, usize),
    }

    let (tx, rx) = mpsc::channel::<Option<GenEvent>>(256);

    let seed = req.seed;
    tokio::task::spawn_blocking(move || {
        let tx2 = tx.clone();
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            crate::generate(
                &prompt_ids, max_new, temperature, top_k, repeat_penalty, seed,
                &cfg, &weights, &gpu, (*tokenizer).as_ref(),
                |piece, is_reasoning| {
                    let _ = tx.blocking_send(Some(GenEvent::Token(piece.to_string(), is_reasoning)));
                },
                |done, total| {
                    let _ = tx2.blocking_send(Some(GenEvent::Prefill(done, total)));
                },
            );
        }));
        if let Err(e) = result {
            let msg = e.downcast_ref::<&str>().copied()
                .or_else(|| e.downcast_ref::<String>().map(|s| s.as_str()))
                .unwrap_or("unknown panic");
            eprintln!("tinyq4: generate panicked: {}", msg);
            let _ = tx2.blocking_send(Some(GenEvent::Token(format!("\n[Error: {}]", msg), false)));
        }
        let _ = tx2.blocking_send(None);
    });

    let stream = ReceiverStream::new(rx).map(|item| -> Result<Event, Infallible> {
        match item {
            Some(GenEvent::Token(piece, is_reasoning)) => {
                let delta = if is_reasoning {
                    serde_json::json!({"reasoning_content": piece})
                } else {
                    serde_json::json!({"content": piece})
                };
                let data = serde_json::json!({
                    "choices": [{"delta": delta, "finish_reason": null}]
                });
                Ok(Event::default().data(data.to_string()))
            }
            Some(GenEvent::Prefill(done, total)) => {
                let data = serde_json::json!({
                    "choices": [{"delta": {"prefill_progress": {"done": done, "total": total}}, "finish_reason": null}]
                });
                Ok(Event::default().data(data.to_string()))
            }
            None => Ok(Event::default().data("[DONE]")),
        }
    });

    Sse::new(stream).keep_alive(KeepAlive::default())
}
