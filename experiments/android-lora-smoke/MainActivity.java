package uk.co.saient.generated.gpt_quartz_lora_smoke;
import android.app.*;
import android.os.*;
import android.content.*;
import android.graphics.Color;
import android.util.Log;
import android.view.*;
import android.widget.*;
import java.io.*;
import java.util.*;

public class MainActivity extends Activity {
 static final int D=8,O=6,R=2,N=96,STEPS=600;
 static final double S=2.0;
 TextView text; Button button; boolean running=false;
 static double[][] arr(int n,int m){return new double[n][m];}
 static double[][] randomMat(Random rng,int n,int m,double sd){double[][] a=arr(n,m);for(int i=0;i<n;i++)for(int j=0;j<m;j++)a[i][j]=rng.nextGaussian()*sd;return a;}
 static double[] forward(double[][] w,double[][] a,double[][] b,double[] x){
  double[] z=new double[R],y=new double[O];
  for(int r=0;r<R;r++)for(int d=0;d<D;d++)z[r]+=a[r][d]*x[d];
  for(int o=0;o<O;o++){for(int d=0;d<D;d++)y[o]+=w[o][d]*x[d];for(int r=0;r<R;r++)y[o]+=S*b[o][r]*z[r];}
  return y;
 }
 static double loss(double[][] w,double[][] a,double[][] b,double[] x,double[] y){
  double[] p=forward(w,a,b,x);double l=0;
  for(int o=0;o<O;o++)l+=.5*(p[o]-y[o])*(p[o]-y[o]);return l;
 }
 static class G {double[][] ga=arr(R,D),gb=arr(O,R);double[] gx=new double[D];}
 static G backward(double[][] w,double[][] a,double[][] b,double[] x,double[] y){
  G g=new G();double[] z=new double[R],p=forward(w,a,b,x),e=new double[O];
  for(int r=0;r<R;r++)for(int d=0;d<D;d++)z[r]+=a[r][d]*x[d];
  for(int o=0;o<O;o++){e[o]=p[o]-y[o];for(int r=0;r<R;r++)g.gb[o][r]=S*e[o]*z[r];for(int d=0;d<D;d++)g.gx[d]+=w[o][d]*e[o];}
  for(int r=0;r<R;r++){double dz=0;for(int o=0;o<O;o++)dz+=S*b[o][r]*e[o];
   for(int d=0;d<D;d++){g.ga[r][d]=dz*x[d];g.gx[d]+=dz*a[r][d];}
  }return g;
 }
 static double rel(double x,double y){return Math.abs(x-y)/Math.max(1e-6,Math.abs(x)+Math.abs(y));}
 static double gradcheck(double[][] w){
  Random rng=new Random(21);double[][] a=randomMat(rng,R,D,.15),b=randomMat(rng,O,R,.14);
  double[] x=new double[D],y=new double[O];for(int d=0;d<D;d++)x[d]=rng.nextGaussian();for(int o=0;o<O;o++)y[o]=rng.nextGaussian();
  G g=backward(w,a,b,x,y);double ep=1e-5,max=0;
  for(int r=0;r<R;r++)for(int d=0;d<D;d++){a[r][d]+=ep;double hi=loss(w,a,b,x,y);a[r][d]-=2*ep;double lo=loss(w,a,b,x,y);a[r][d]+=ep;max=Math.max(max,rel((hi-lo)/(2*ep),g.ga[r][d]));}
  for(int o=0;o<O;o++)for(int r=0;r<R;r++){b[o][r]+=ep;double hi=loss(w,a,b,x,y);b[o][r]-=2*ep;double lo=loss(w,a,b,x,y);b[o][r]+=ep;max=Math.max(max,rel((hi-lo)/(2*ep),g.gb[o][r]));}
  for(int d=0;d<D;d++){x[d]+=ep;double hi=loss(w,a,b,x,y);x[d]-=2*ep;double lo=loss(w,a,b,x,y);x[d]+=ep;max=Math.max(max,rel((hi-lo)/(2*ep),g.gx[d]));}
  return max;
 }
 static long hash(double[][] w){long h=1469598103934665603L;for(double[] row:w)for(double v:row)h=h*31+Double.doubleToLongBits(v);return h;}
 static double eval(double[][] w,double[][] a,double[][] b,double[][] x,double[][] y){double l=0;for(int i=0;i<x.length;i++)l+=loss(w,a,b,x[i],y[i]);return l/x.length;}
 static void update(double[][] p,double[][] g,double[][] m,double[][] v,int t,int batch){
  double beta1=.9,beta2=.999;
  for(int i=0;i<p.length;i++)for(int j=0;j<p[i].length;j++){
   double d=g[i][j]/batch;m[i][j]=beta1*m[i][j]+(1-beta1)*d;v[i][j]=beta2*v[i][j]+(1-beta2)*d*d;
   p[i][j]-=.03*(m[i][j]/(1-Math.pow(beta1,t)))/(Math.sqrt(v[i][j]/(1-Math.pow(beta2,t)))+1e-8);
  }
 }
 String test()throws Exception{
  long start=SystemClock.elapsedRealtime();Random rng=new Random(20261028);
  double[][] w=randomMat(rng,O,D,.3);long frozenHash=hash(w);double gradientError=gradcheck(w);
  if(gradientError>1e-4)throw new IllegalStateException("Gradient mismatch "+gradientError);
  double[][] teacherA=randomMat(rng,R,D,.3),teacherB=randomMat(rng,O,R,.3);
  double[][] x=arr(N,D),y=arr(N,O),xt=arr(48,D),yt=arr(48,O);
  for(int i=0;i<x.length;i++){for(int d=0;d<D;d++)x[i][d]=rng.nextGaussian();y[i]=forward(w,teacherA,teacherB,x[i]);}
  for(int i=0;i<xt.length;i++){for(int d=0;d<D;d++)xt[i][d]=rng.nextGaussian();yt[i]=forward(w,teacherA,teacherB,xt[i]);}
  double[][] a=randomMat(rng,R,D,.12),b=arr(O,R),ma=arr(R,D),mb=arr(O,R),va=arr(R,D),vb=arr(O,R);
  double initial=eval(w,a,b,xt,yt);
  for(int step=1;step<=STEPS;step++){
   double[][] ga=arr(R,D),gb=arr(O,R);
   for(int i=0;i<N;i++){G g=backward(w,a,b,x[i],y[i]);for(int r=0;r<R;r++)for(int d=0;d<D;d++)ga[r][d]+=g.ga[r][d];for(int o=0;o<O;o++)for(int r=0;r<R;r++)gb[o][r]+=g.gb[o][r];}
   update(a,ga,ma,va,step,N);update(b,gb,mb,vb,step,N);
   if(step%150==0)Log.i("QuartzLoRA","step="+step+" loss="+eval(w,a,b,x,y));
  }
  double finalLoss=eval(w,a,b,xt,yt),trainingLoss=eval(w,a,b,x,y);
  boolean frozen=(frozenHash==hash(w));
  File f=new File(getFilesDir(),"synthetic-lora-adapter.txt");
  Writer out=new OutputStreamWriter(new FileOutputStream(f),"UTF-8");
  try{out.write("synthetic-test-only; D=8 O=6 rank=2 alpha=4\nbase_hash="+Long.toHexString(frozenHash)+"\n");
   for(int r=0;r<R;r++)for(int d=0;d<D;d++)out.write("A "+r+" "+d+" "+a[r][d]+"\n");
   for(int o=0;o<O;o++)for(int r=0;r<R;r++)out.write("B "+o+" "+r+" "+b[o][r]+"\n");
  }finally{out.close();}
  PowerManager pm=(PowerManager)getSystemService(Context.POWER_SERVICE);int thermal=pm==null?-1:pm.getCurrentThermalStatus();
  boolean ok=frozen&&finalLoss<initial*.01&&gradientError<1e-4;
  return String.format(Locale.US,"RESULT: %s\n\nGradient finite-difference max error: %.2g\nBase weights unchanged: %s\nHeld-out loss: %.6f → %.9f\nTrain loss: %.9f\nOptimizer steps: %d\nDuration: %d ms\nThermal status: %d\nAdapter saved privately: %s\n\nREAL on-device LoRA training on a synthetic linear layer. Not yet GGUF transformer fine-tuning.",ok?"PASS":"FAIL",gradientError,frozen?"YES":"NO",initial,finalLoss,trainingLoss,STEPS,SystemClock.elapsedRealtime()-start,thermal,f.getAbsolutePath());
 }
 void launch(){if(running)return;running=true;button.setEnabled(false);text.setText("Training locally on the Samsung...");new Thread(new Runnable(){public void run(){String r;try{r=test();Log.i("QuartzLoRA",r);}catch(Throwable ex){r="FAIL "+ex;Log.e("QuartzLoRA",r,ex);}final String res=r;runOnUiThread(new Runnable(){public void run(){text.setText(res);running=false;button.setEnabled(true);}});}}).start();}
 public void onCreate(Bundle bundle){super.onCreate(bundle);ScrollView scroll=new ScrollView(this);LinearLayout box=new LinearLayout(this);box.setPadding(32,110,32,48);box.setOrientation(1);box.setBackgroundColor(Color.rgb(13,21,32));scroll.addView(box);TextView title=new TextView(this);title.setText("Quartz LoRA · Samsung");title.setTextColor(Color.WHITE);title.setTextSize(24);box.addView(title);TextView description=new TextView(this);description.setText("Gradient + Adam training smoke test\nSynthetic 6×8 layer | rank 2 | frozen base");description.setTextColor(Color.rgb(100,230,190));box.addView(description);button=new Button(this);button.setText("Run training again");box.addView(button);text=new TextView(this);text.setTextColor(Color.WHITE);text.setTextSize(17);box.addView(text);button.setOnClickListener(new View.OnClickListener(){public void onClick(View v){launch();}});setContentView(scroll);launch();}
}