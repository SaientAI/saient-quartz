# ComfyUI power without the Python dependency maze

**Run local AI without building your workflow around Python environments,
custom-node dependency chains and manual model plumbing.**

[Saient](https://saient.co.uk) is a native local AI workspace built around the
[Quartz engine](https://github.com/SaientAI/saient-quartz), a Rust inference
runtime designed to execute AI workloads directly on your hardware.

Saient brings image generation, video generation, local language models and AI
agents into one environment while managing more of the machinery underneath.

**Local models. Native execution. One workspace.**

> Looking for the product-level feature comparison? Read
> **[Saient vs ComfyUI: a task-focused local AI alternative](https://saient.co.uk/comfyui-alternative)**.

## Built differently from the bottom up

ComfyUI is incredibly powerful, but that power comes from an ecosystem of
Python packages, PyTorch builds, custom nodes and community extensions.

Saient takes a different approach. Instead of adding another interface on top
of the traditional local-AI stack, development started further down.

**Quartz is the native execution layer. Saient is the system built around it.**

That gives the project control over how models are loaded, how memory is
managed, how hardware is accessed and how different AI workloads interact.

### ComfyUI: you build the workflow

Models, nodes, extensions, dependencies and execution graphs are exposed
directly to the user. That flexibility is exactly why ComfyUI is so powerful.

### Saient: you describe what you want to do

Saient is being built to manage the models, runtime, tools and workflow needed
to perform a task. The aim is not to recreate a node graph with prettier
buttons. It is to make direct operation of the graph unnecessary for everyday
use.

This is a product direction, not a claim of feature parity. ComfyUI remains the
stronger choice when you need visual graph composition, its custom-node
ecosystem, reusable workflow JSON or model support Saient has not implemented.

## From LLMs to SDXL to Wan

Local AI has traditionally been split into separate ecosystems: one
application for LLMs, another for Stable Diffusion, another stack for video
generation and another framework for agents.

Saient is built around the idea that these are different capabilities
available to the same system. Quartz currently supplies native LLM and image
execution. Saient coordinates those capabilities with its agent environment
and video tooling; the shipped Wan path currently uses a separate companion
engine while a native Quartz Wan implementation is developed and optimized.

The longer-term workflow can move naturally between:

**reasoning → image generation → image analysis → video generation → verification**

without requiring the user to rebuild the pipeline manually each time.

## Desktop AI architecture that also runs on phones

Quartz is not limited to conventional desktop inference. Development includes
Vulkan execution on Android, and Saient has run Wan2.1 1.3B locally on a
Samsung smartphone through its on-device video engine.

That work demonstrates what the system is being designed around:

- local execution
- constrained hardware
- portable inference
- direct hardware access
- reduced dependence on heavyweight external runtimes

The repository README documents the exact boundary between Quartz and the
current Wan companion engine, along with the measured status of the native Rust
Wan implementation.

## No cloud required for the core experience

Saient is designed as a local-first system. Models and workspace data can stay
on the user's machine. After the required models and components are installed,
the core local workflows are designed to remain usable without an internet
connection.

Hosted models can still be used when wanted. They do not have to own the
system.

## Where GPT fits

Saient does not treat GPT, Qwen, DeepSeek or another language model as the
application itself. The model is a capability within the workspace.

A hosted model can provide capable reasoning when internet access is available.
A local model can take its place when a fully offline system is preferred. The
surrounding workspace, tools and execution model remain Saient.

**The LLM can change. The system remains Saient.**

## Not another ComfyUI skin

Saient is not trying to rebuild ComfyUI node-for-node. ComfyUI already does
that job exceptionally well.

Saient is exploring the other direction:

**What happens when the user no longer has to operate the inference graph at
all?**

Instead of:

`load model → connect nodes → configure sampler → execute → inspect → modify → execute again`

the intended experience is closer to:

**Tell Saient what you want. Let the system work out how to execute it.**

## Saient + Quartz

**Native local AI without making the runtime your full-time hobby.**

LLMs. Agents. SDXL. Wan. Desktop. Android. Local-first. One system.

- [Download and learn about Saient](https://saient.co.uk)
- [Read the direct Saient vs ComfyUI comparison](https://saient.co.uk/comfyui-alternative)
- [Watch Wan2.1 generate locally on a phone](https://youtu.be/ZXc5hBgoro0)
