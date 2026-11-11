// FastConformer/TDT for the pinned Parakeet model. All neural arithmetic stays
// on CUDA; Rust owns only the shared greedy policy and wire observations.
#include "cuda_family_common.cuh"
#include <cublas_v2.h>

namespace {
using synapse_cuda_family::DeviceAllocation;
using Buffer = DeviceAllocation<float>;
using HalfBuffer = DeviceAllocation<half>;
thread_local std::string error;
constexpr int H = 1024, D = 640, HEADS = 8, HD = 128, TILE = 128;

__global__ void to_half(const float *x,half *y,int n) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n)y[i]=__float2half_rn(x[i]);
}

__global__ void bias(float *x, const float *b, int n, int width, bool relu) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) { float v = x[i] + (b ? b[i % width] : 0.f); x[i] = relu ? fmaxf(v, 0.f) : v; }
}
__global__ void silu(float *x, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] = x[i] / (1.f + expf(-x[i]));
}
__global__ void residual(float *x, const float *y, int n, float scale) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] += scale * y[i];
}
__global__ void norm(const float *x, float *y, const float *w, const float *b, int rows) {
    int row = blockIdx.x, tid = threadIdx.x;
    __shared__ float sums[256];
    float sum = 0.f;
    for (int j = tid; j < H; j += blockDim.x) sum += x[row * H + j];
    sums[tid] = sum; __syncthreads();
    for (int s = 128; s; s >>= 1) { if (tid < s) sums[tid] += sums[tid+s]; __syncthreads(); }
    float mean = sums[0] / H, var = 0.f;
    // Every warp must capture the mean before any warp reuses sums for variance.
    __syncthreads();
    for (int j = tid; j < H; j += blockDim.x) { float d = x[row * H+j]-mean; var += d*d; }
    sums[tid] = var; __syncthreads();
    for (int s = 128; s; s >>= 1) { if (tid < s) sums[tid] += sums[tid+s]; __syncthreads(); }
    float inv = rsqrtf(sums[0] / H + 1e-5f);
    for (int j = tid; j < H; j += blockDim.x) y[row*H+j] = (x[row*H+j]-mean)*inv*w[j]+b[j];
}

// Time tiles include the exact receptive-field halo. No full-resolution
// 256-channel subsampling activation scales with recording duration.
__global__ void conv2d(const float *x, float *y, const float *w, const float *b,
                       int in_start, int in_rows, int in_freq, int out_start,
                       int out_rows, int valid_rows, int actual_rows, bool first) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int out_freq = (in_freq+1)/2, n = out_rows*out_freq*256;
    if (i >= n) return;
    int c = i%256, f = (i/256)%out_freq, t = out_start + i/(256*out_freq);
    if (t < 0 || t >= valid_rows || t >= actual_rows) { y[i]=0.f; return; }
    float sum = b[c];
    for (int kt=0; kt<3; ++kt) for (int kf=0; kf<3; ++kf) {
        int it=2*t+kt-1-in_start, jf=2*f+kf-1;
        if (it>=0 && it<in_rows && jf>=0 && jf<in_freq)
            sum += x[(it*in_freq+jf)*(first?1:256)+(first?0:c)] * w[c*9+kt*3+kf];
    }
    y[i] = first ? fmaxf(sum,0.f) : sum;
}
__global__ void mask_sub(float *x, int rows, int freq, int start, int valid, int actual,
                         const float *b) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if (i<rows*freq*256) {
        int t=start+i/(freq*256);
        x[i]=(t<0 || t>=valid || t>=actual) ? 0.f : fmaxf(x[i]+b[i%256],0.f);
    }
}
__global__ void flatten_sub(const float *x, float *y, int rows) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if (i<rows*4096) {
        int f=i%16, c=(i/16)%256, t=i/4096;
        y[i]=x[(t*16+f)*256+c];
    }
}
__global__ void positions(float *x, int t) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if (i<(2*t-1)*H) {
        int j=i%H, p=t-1-i/H;
        float angle=p*powf(10000.f,-float((j/2)*2)/H);
        x[i]=(j%2) ? cosf(angle) : sinf(angle);
    }
}
__global__ void query_bias(const float *q, const float *b, float *out, int rows, int head, int start) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if (i<rows*HD) out[i]=q[(start+i/HD)*H+head*HD+i%HD]+b[head*HD+i%HD];
}
__global__ void attention_softmax(float *ac, const float *bd, int rows, int t, int start, int valid) {
    int row=blockIdx.x, tid=threadIdx.x;
    __shared__ float sums[256];
    float m=-INFINITY;
    for (int j=tid;j<t;j+=256) {
        float v=(start+row<valid && j<valid)
            ? (ac[row*t+j]+bd[row*(2*t-1)+(t-1-(start+row)+j)]) * 0.08838834764831845f
            : -10000.f;
        ac[row*t+j]=v; m=fmaxf(m,v);
    }
    sums[tid]=m; __syncthreads();
    for (int s=128;s;s>>=1) { if(tid<s) sums[tid]=fmaxf(sums[tid],sums[tid+s]); __syncthreads(); }
    m=sums[0]; float sum=0.f;
    // Every warp must capture the maximum before any warp reuses sums for totals.
    __syncthreads();
    for (int j=tid;j<t;j+=256) { float v=expf(ac[row*t+j]-m); ac[row*t+j]=v; sum+=v; }
    sums[tid]=sum; __syncthreads();
    for(int s=128;s;s>>=1) { if(tid<s) sums[tid]+=sums[tid+s]; __syncthreads(); }
    for(int j=tid;j<t;j+=256)
        ac[row*t+j]=(start+row<valid && j<valid) ? ac[row*t+j]/sums[0] : 0.f;
}
__global__ void glu(const float *x,float *y,int t,int valid) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i<t*H) y[i]=(i/H<valid) ? x[(i/H)*2*H+i%H]/(1.f+expf(-x[(i/H)*2*H+H+i%H])) : 0.f;
}
__global__ void depthwise(const float *x,float *y,const float *w,const float *mean,
                          const float *var,const float *scale,const float *b,int t) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=t*H) return;
    int c=i%H,r=i/H; float v=0.f;
    for(int k=0;k<9;++k) if(r+k-4>=0 && r+k-4<t) v+=x[(r+k-4)*H+c]*w[c*9+k];
    v=(v-mean[c])*rsqrtf(var[c]+1e-5f)*scale[c]+b[c];
    y[i]=v/(1.f+expf(-v));
}
__global__ void lstm_update(float *h,float *c,const float *a,const float *b,
                           const float *bi,const float *bh) {
    int j=blockIdx.x*blockDim.x+threadIdx.x;
    if(j>=D) return;
    float g[4]; for(int k=0;k<4;++k) g[k]=a[k*D+j]+b[k*D+j]+bi[k*D+j]+bh[k*D+j];
    float cell=1.f/(1.f+expf(-g[1]))*c[j]+1.f/(1.f+expf(-g[0]))*tanhf(g[2]);
    c[j]=cell; h[j]=1.f/(1.f+expf(-g[3]))*tanhf(cell);
}
__global__ void joint_input(const float *enc,const float *pred,float *out,int frame) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i<D) out[i]=fmaxf(enc[frame*D+i]+pred[i],0.f);
}

struct Context {
    cublasHandle_t blas=nullptr;
    bool strict_f32=false;
    int device=0, frames=0, valid=0, next_layer=0;
    size_t peak=0;
    std::unordered_map<std::string,Buffer> weights;
    std::unordered_map<std::string,HalfBuffer> half_weights;
    Buffer hidden, pos, projected, h, cell, predictor;
    explicit Context(bool fp32=false):strict_f32(fp32) {
        FAMILY_CUDA_CHECK(cudaGetDevice(&device));
        FAMILY_CUBLAS_CHECK(cublasCreate(&blas));
        FAMILY_CUBLAS_CHECK(cublasSetMathMode(blas,CUBLAS_PEDANTIC_MATH));
        FAMILY_CUBLAS_CHECK(cublasSetAtomicsMode(blas,CUBLAS_ATOMICS_NOT_ALLOWED));
    }
    ~Context(){ if(blas) cublasDestroy(blas); }
    void bind(){ FAMILY_CUDA_CHECK(cudaSetDevice(device)); }
    float *w(const std::string &name){ return weights.at(name).pointer; }
    void sync(){ FAMILY_CUDA_CHECK(cudaGetLastError()); FAMILY_CUDA_CHECK(cudaDeviceSynchronize()); }
    void memory(){ size_t free,total; FAMILY_CUDA_CHECK(cudaMemGetInfo(&free,&total)); peak=std::max(peak,total-free); }
    void multiply(const float *x,float *y,const std::string &name,int rows,int in,int out) {
        float one=1.f,zero=0.f;
        if(strict_f32) {
            FAMILY_CUBLAS_CHECK(cublasGemmEx(blas,CUBLAS_OP_T,CUBLAS_OP_N,out,rows,in,&one,
                w(name),CUDA_R_32F,in,x,CUDA_R_32F,in,&zero,y,CUDA_R_32F,out,
                CUBLAS_COMPUTE_32F_PEDANTIC,CUBLAS_GEMM_DEFAULT));
            return;
        }
        HalfBuffer input(rows*in);
        to_half<<<(rows*in+255)/256,256>>>(x,input.pointer,rows*in);
        FAMILY_CUBLAS_CHECK(cublasGemmEx(blas,CUBLAS_OP_T,CUBLAS_OP_N,out,rows,in,&one,
            half_weights.at(name).pointer,CUDA_R_16F,in,input.pointer,CUDA_R_16F,in,&zero,y,CUDA_R_32F,out,
            CUBLAS_COMPUTE_32F,CUBLAS_GEMM_DEFAULT));
    }
    void linear(const float *x,float *y,const std::string &name,int rows,int in,int out,bool has_bias=false) {
        multiply(x,y,name+".weight",rows,in,out);
        if(has_bias) bias<<<(rows*out+255)/256,256>>>(y,w(name+".bias"),rows*out,out,false);
    }
    void ln(const float *x,float *y,const std::string &name) {
        norm<<<frames,256>>>(x,y,w(name+".weight"),w(name+".bias"),frames);
    }
    void feedforward(const std::string &l,int index) {
        Buffer n(frames*H),a(frames*4096),b(frames*H);
        std::string ff=l+".feed_forward"+std::to_string(index);
        ln(hidden.pointer,n.pointer,l+".norm_feed_forward"+std::to_string(index));
        linear(n.pointer,a.pointer,ff+".linear1",frames,H,4096);
        silu<<<(frames*4096+255)/256,256>>>(a.pointer,frames*4096);
        linear(a.pointer,b.pointer,ff+".linear2",frames,4096,H);
        residual<<<(frames*H+255)/256,256>>>(hidden.pointer,b.pointer,frames*H,.5f);
        memory();
    }
};

template<typename F> int guard(F f) {
    try { f(); error.clear(); return 0; }
    catch(const std::exception &e){ error=e.what(); return -1; }
}
} // namespace

extern "C" const char *synapse_parakeet_error(){ return error.c_str(); }
extern "C" void *synapse_parakeet_create(){
    try { return new Context(); } catch(const std::exception &e){error=e.what();return nullptr;}
}
extern "C" void *synapse_parakeet_create_f32(){
    try { return new Context(true); } catch(const std::exception &e){error=e.what();return nullptr;}
}
extern "C" void synapse_parakeet_destroy(void *p){ if(p){auto *c=static_cast<Context*>(p); cudaSetDevice(c->device); delete c;} }
extern "C" int synapse_parakeet_weight(void *p,const char *name,const float *data,size_t n){
    return guard([&]{ auto &c=*static_cast<Context*>(p);c.bind(); Buffer b(n);
        FAMILY_CUDA_CHECK(cudaMemcpy(b.pointer,data,n*sizeof(float),cudaMemcpyHostToDevice));
        // The f16 path retains a float view for scalar/direct-convolution consumers.
        // Strict f32 keeps only the float view and never rounds weights through half.
        if(!c.strict_f32) {
            HalfBuffer hb(n);to_half<<<(n+255)/256,256>>>(b.pointer,hb.pointer,static_cast<int>(n));
            c.half_weights.emplace(name,std::move(hb));
        }
        c.weights.emplace(name,std::move(b));c.memory(); });
}

extern "C" int synapse_parakeet_begin(void *p,const float *mel,int mel_rows,int valid_mel,int rung){
    return guard([&]{
        auto &c=*static_cast<Context*>(p);c.bind();
        c.frames=rung;c.valid=(valid_mel+7)/8;c.next_layer=0;
        c.projected.reset();c.h.reset();c.cell.reset();c.predictor.reset();
        c.hidden.allocate(rung*H);c.pos.allocate((2*rung-1)*H);
        FAMILY_CUDA_CHECK(cudaMemset(c.hidden.pointer,0,rung*H*sizeof(float)));
        Buffer input(mel_rows*128);
        FAMILY_CUDA_CHECK(cudaMemcpy(input.pointer,mel,mel_rows*128*sizeof(float),cudaMemcpyHostToDevice));
        int a1=(mel_rows+1)/2,a2=(a1+1)/2,a3=(a2+1)/2;
        int v1=(valid_mel+1)/2,v2=(v1+1)/2,v3=(v2+1)/2;
        const std::string sub="encoder.subsampling.";
        for(int start=0;start<rung;start+=TILE){
            int rows=std::min(TILE,rung-start),s1=4*start-3,n1=4*rows+3,s2=2*start-1,n2=2*rows+1;
            Buffer a(n1*64*256),b(n2*32*256),d(n2*32*256),e(rows*16*256),f(rows*16*256),flat(rows*4096);
            conv2d<<<(n1*64*256+255)/256,256>>>(input.pointer,a.pointer,c.w(sub+"layers.0.weight"),
                c.w(sub+"layers.0.bias"),0,mel_rows,128,s1,n1,v1,a1,true);
            conv2d<<<(n2*32*256+255)/256,256>>>(a.pointer,b.pointer,c.w(sub+"layers.2.weight"),
                c.w(sub+"layers.2.bias"),s1,n1,64,s2,n2,v2,a2,false);
            c.linear(b.pointer,d.pointer,sub+"layers.3",n2*32,256,256);
            mask_sub<<<(n2*32*256+255)/256,256>>>(d.pointer,n2,32,s2,v2,a2,c.w(sub+"layers.3.bias"));
            conv2d<<<(rows*16*256+255)/256,256>>>(d.pointer,e.pointer,c.w(sub+"layers.5.weight"),
                c.w(sub+"layers.5.bias"),s2,n2,32,start,rows,v3,a3,false);
            c.linear(e.pointer,f.pointer,sub+"layers.6",rows*16,256,256);
            mask_sub<<<(rows*16*256+255)/256,256>>>(f.pointer,rows,16,start,v3,a3,c.w(sub+"layers.6.bias"));
            flatten_sub<<<(rows*4096+255)/256,256>>>(f.pointer,flat.pointer,rows);
            c.linear(flat.pointer,c.hidden.pointer+start*H,sub+"linear",rows,4096,H,true);
            c.memory();
        }
        positions<<<((2*rung-1)*H+255)/256,256>>>(c.pos.pointer,rung);
        c.sync();c.memory();
    });
}

extern "C" int synapse_parakeet_layer(void *p,int layer){
    return guard([&]{
        auto &c=*static_cast<Context*>(p);c.bind();
        if(layer!=c.next_layer || layer>=24) throw std::runtime_error("encoder layer order");
        int t=c.frames;std::string l="encoder.layers."+std::to_string(layer);
        c.feedforward(l,1);
        {
            Buffer n(t*H),q(t*H),k(t*H),v(t*H),r((2*t-1)*H),att(t*H),out(t*H);
            Buffer qu(TILE*HD),qv(TILE*HD),ac(TILE*t),bd(TILE*(2*t-1));
            HalfBuffer kh(c.strict_f32?0:t*H),vh(c.strict_f32?0:t*H),rh(c.strict_f32?0:(2*t-1)*H),
                quh(c.strict_f32?0:TILE*HD),qvh(c.strict_f32?0:TILE*HD),prob(c.strict_f32?0:TILE*t);
            c.ln(c.hidden.pointer,n.pointer,l+".norm_self_att");
            std::string a=l+".self_attn";
            c.linear(n.pointer,q.pointer,a+".q_proj",t,H,H);
            c.linear(n.pointer,k.pointer,a+".k_proj",t,H,H);
            c.linear(n.pointer,v.pointer,a+".v_proj",t,H,H);
            c.linear(c.pos.pointer,r.pointer,a+".relative_k_proj",2*t-1,H,H);
            if(!c.strict_f32) {
                to_half<<<(t*H+255)/256,256>>>(k.pointer,kh.pointer,t*H);
                to_half<<<(t*H+255)/256,256>>>(v.pointer,vh.pointer,t*H);
                to_half<<<((2*t-1)*H+255)/256,256>>>(r.pointer,rh.pointer,(2*t-1)*H);
            }
            float one=1.f,zero=0.f;
            for(int head=0;head<HEADS;++head) for(int start=0;start<t;start+=TILE){
                int rows=std::min(TILE,t-start);
                query_bias<<<(rows*HD+255)/256,256>>>(q.pointer,c.w(a+".bias_u"),qu.pointer,rows,head,start);
                query_bias<<<(rows*HD+255)/256,256>>>(q.pointer,c.w(a+".bias_v"),qv.pointer,rows,head,start);
                if(c.strict_f32) {
                    FAMILY_CUBLAS_CHECK(cublasGemmEx(c.blas,CUBLAS_OP_T,CUBLAS_OP_N,t,rows,HD,&one,
                        k.pointer+head*HD,CUDA_R_32F,H,qu.pointer,CUDA_R_32F,HD,&zero,ac.pointer,CUDA_R_32F,t,
                        CUBLAS_COMPUTE_32F_PEDANTIC,CUBLAS_GEMM_DEFAULT));
                    FAMILY_CUBLAS_CHECK(cublasGemmEx(c.blas,CUBLAS_OP_T,CUBLAS_OP_N,2*t-1,rows,HD,&one,
                        r.pointer+head*HD,CUDA_R_32F,H,qv.pointer,CUDA_R_32F,HD,&zero,bd.pointer,CUDA_R_32F,2*t-1,
                        CUBLAS_COMPUTE_32F_PEDANTIC,CUBLAS_GEMM_DEFAULT));
                    attention_softmax<<<rows,256>>>(ac.pointer,bd.pointer,rows,t,start,c.valid);
                    FAMILY_CUBLAS_CHECK(cublasGemmEx(c.blas,CUBLAS_OP_N,CUBLAS_OP_N,HD,rows,t,&one,
                        v.pointer+head*HD,CUDA_R_32F,H,ac.pointer,CUDA_R_32F,t,&zero,att.pointer+start*H+head*HD,CUDA_R_32F,H,
                        CUBLAS_COMPUTE_32F_PEDANTIC,CUBLAS_GEMM_DEFAULT));
                    continue;
                }
                to_half<<<(rows*HD+255)/256,256>>>(qu.pointer,quh.pointer,rows*HD);
                to_half<<<(rows*HD+255)/256,256>>>(qv.pointer,qvh.pointer,rows*HD);
                // Row-major scores: Q K^T and Q R^T. Relative shift selects
                // position t-1-query+key without materializing a shifted matrix.
                FAMILY_CUBLAS_CHECK(cublasGemmEx(c.blas,CUBLAS_OP_T,CUBLAS_OP_N,t,rows,HD,&one,
                    kh.pointer+head*HD,CUDA_R_16F,H,quh.pointer,CUDA_R_16F,HD,&zero,ac.pointer,CUDA_R_32F,t,
                    CUBLAS_COMPUTE_32F,CUBLAS_GEMM_DEFAULT));
                FAMILY_CUBLAS_CHECK(cublasGemmEx(c.blas,CUBLAS_OP_T,CUBLAS_OP_N,2*t-1,rows,HD,&one,
                    rh.pointer+head*HD,CUDA_R_16F,H,qvh.pointer,CUDA_R_16F,HD,&zero,bd.pointer,CUDA_R_32F,2*t-1,
                    CUBLAS_COMPUTE_32F,CUBLAS_GEMM_DEFAULT));
                attention_softmax<<<rows,256>>>(ac.pointer,bd.pointer,rows,t,start,c.valid);
                to_half<<<(rows*t+255)/256,256>>>(ac.pointer,prob.pointer,rows*t);
                FAMILY_CUBLAS_CHECK(cublasGemmEx(c.blas,CUBLAS_OP_N,CUBLAS_OP_N,HD,rows,t,&one,
                    vh.pointer+head*HD,CUDA_R_16F,H,prob.pointer,CUDA_R_16F,t,&zero,att.pointer+start*H+head*HD,CUDA_R_32F,H,
                    CUBLAS_COMPUTE_32F,CUBLAS_GEMM_DEFAULT));
            }
            c.linear(att.pointer,out.pointer,a+".o_proj",t,H,H);
            residual<<<(t*H+255)/256,256>>>(c.hidden.pointer,out.pointer,t*H,1.f);c.memory();
        }
        {
            Buffer n(t*H),a(t*2*H),g(t*H),d(t*H),out(t*H);
            c.ln(c.hidden.pointer,n.pointer,l+".norm_conv");
            std::string cv=l+".conv";
            c.linear(n.pointer,a.pointer,cv+".pointwise_conv1",t,H,2*H);
            glu<<<(t*H+255)/256,256>>>(a.pointer,g.pointer,t,c.valid);
            depthwise<<<(t*H+255)/256,256>>>(g.pointer,d.pointer,c.w(cv+".depthwise_conv.weight"),
                c.w(cv+".norm.running_mean"),c.w(cv+".norm.running_var"),c.w(cv+".norm.weight"),c.w(cv+".norm.bias"),t);
            c.linear(d.pointer,out.pointer,cv+".pointwise_conv2",t,H,H);
            residual<<<(t*H+255)/256,256>>>(c.hidden.pointer,out.pointer,t*H,1.f);c.memory();
        }
        c.feedforward(l,2);
        Buffer normalized(t*H);c.ln(c.hidden.pointer,normalized.pointer,l+".norm_out");
        c.hidden=std::move(normalized);c.sync();++c.next_layer;
    });
}

extern "C" int synapse_parakeet_finish(void *p,float *encoder){
    return guard([&]{ auto &c=*static_cast<Context*>(p);c.bind();
        if(c.next_layer!=24) throw std::runtime_error("incomplete encoder");
        c.projected.allocate(c.valid*D);
        c.linear(c.hidden.pointer,c.projected.pointer,"encoder_projector",c.valid,H,D,true);
        if(encoder) FAMILY_CUDA_CHECK(cudaMemcpy(encoder,c.hidden.pointer,c.valid*H*sizeof(float),cudaMemcpyDeviceToHost));
        c.sync();c.memory();c.hidden.reset();c.pos.reset();
    });
}

static void predict(Context &c,int token){
    Buffer a(4*D),b(4*D);
    const float *input=c.w("decoder.embedding.weight")+token*D;
    for(int layer=0;layer<2;++layer){
        std::string base="decoder.lstm.",suffix="_l"+std::to_string(layer);
        c.multiply(input,a.pointer,base+"weight_ih"+suffix,1,D,4*D);
        c.multiply(c.h.pointer+layer*D,b.pointer,base+"weight_hh"+suffix,1,D,4*D);
        lstm_update<<<(D+255)/256,256>>>(c.h.pointer+layer*D,c.cell.pointer+layer*D,a.pointer,b.pointer,
            c.w(base+"bias_ih"+suffix),c.w(base+"bias_hh"+suffix));
        input=c.h.pointer+layer*D;
    }
    c.linear(input,c.predictor.pointer,"decoder.decoder_projector",1,D,D,true);
}
extern "C" int synapse_parakeet_predict(void *p,int token,int reset){
    return guard([&]{ auto &c=*static_cast<Context*>(p);c.bind();
        if(reset){ c.h.allocate(2*D);c.cell.allocate(2*D);c.predictor.allocate(D);
            FAMILY_CUDA_CHECK(cudaMemset(c.h.pointer,0,2*D*sizeof(float)));
            FAMILY_CUDA_CHECK(cudaMemset(c.cell.pointer,0,2*D*sizeof(float))); }
        predict(c,token);c.sync();c.memory(); });
}
extern "C" int synapse_parakeet_joint(void *p,int frame,float *logits){
    return guard([&]{ auto &c=*static_cast<Context*>(p);c.bind();
        if(frame<0 || frame>=c.valid) throw std::runtime_error("invalid encoder frame");
        Buffer input(D),out(8198);
        joint_input<<<(D+255)/256,256>>>(c.projected.pointer,c.predictor.pointer,input.pointer,frame);
        c.linear(input.pointer,out.pointer,"joint.head",1,D,8198,true);
        FAMILY_CUDA_CHECK(cudaMemcpy(logits,out.pointer,8198*sizeof(float),cudaMemcpyDeviceToHost));c.memory(); });
}
extern "C" size_t synapse_parakeet_peak(void *p){return static_cast<Context*>(p)->peak;}
