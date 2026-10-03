#extension GL_EXT_shader_16bit_storage : require
#extension GL_EXT_shader_explicit_arithmetic_types_float16 : require
layout(local_size_x=64) in;
layout(set=0,binding=0,std430) readonly buffer Input { float x[]; };
layout(set=0,binding=1,std430) readonly buffer Weight { float16_t w[]; };
layout(set=0,binding=2,std430) writeonly buffer Output { float y[]; };
layout(set=0,binding=3,std430) readonly buffer Other { float z[]; };
layout(set=0,binding=4,std430) readonly buffer Ids { int ids[]; };
layout(set=0,binding=5,std430) readonly buffer Mask { int mask[]; };
layout(set=0,binding=6,std430) readonly buffer Values { float v[]; };
layout(push_constant) uniform Params {
    uint op; uint rows; uint cols; uint inner;
    uint seq; uint heads; uint kv_heads; uint dim;
    uint flags; uint offset; uint stride; uint window;
    float eps; float theta; float scale; float unused;
} p;
float gelu(float a) {
    // Abramowitz-Stegun erf approximation avoids dependence on GLSL vendor intrinsics.
    float q=abs(a)*0.7071067811865475;
    float t=1.0/(1.0+0.3275911*q);
    float e=1.0-(((((1.061405429*t-1.453152027)*t)+1.421413741)*t-0.284496736)*t+0.254829592)*t*exp(-q*q);
    return 0.5*a*(1.0+sign(a)*e);
}
#ifdef COOPERATIVE
shared float16_t tile_a[4096];
shared float tile_c[4096];
void cooperative_linear() {
    uint tile=(gl_WorkGroupID.y*gl_NumWorkGroups.x+gl_WorkGroupID.x)*gl_NumSubgroups+gl_SubgroupID;
    uint column_tiles=(p.cols+15)/16;
    uint row=tile/column_tiles*16, column=tile%column_tiles*16;
    if(row>=p.rows) return;
    uint base=gl_SubgroupID*256;
    coopmat<float, gl_ScopeSubgroup, 16, 16, gl_MatrixUseAccumulator> c=coopmat<float, gl_ScopeSubgroup, 16, 16, gl_MatrixUseAccumulator>(0.0);
    for(uint k=0;k<p.inner;k+=16) {
        for(uint j=gl_SubgroupInvocationID;j<256;j+=gl_SubgroupSize) {
            uint r=j/16, d=j%16;
            tile_a[base+j]=float16_t((row+r<p.rows && k+d<p.inner)?x[(row+r)*p.inner+k+d]:0.0);
        }
        subgroupMemoryBarrierShared(); subgroupBarrier();
        coopmat<float16_t, gl_ScopeSubgroup, 16, 16, gl_MatrixUseA> a;
        coopmat<float16_t, gl_ScopeSubgroup, 16, 16, gl_MatrixUseB> b;
        coopMatLoad(a,tile_a,base,16,gl_CooperativeMatrixLayoutRowMajor);
        coopMatLoad(b,w,(column+p.offset)*p.inner+k,p.inner,gl_CooperativeMatrixLayoutColumnMajor);
        c=coopMatMulAdd(a,b,c);
        subgroupBarrier();
    }
    coopMatStore(c,tile_c,base,16,gl_CooperativeMatrixLayoutRowMajor);
    subgroupMemoryBarrierShared(); subgroupBarrier();
    for(uint j=gl_SubgroupInvocationID;j<256;j+=gl_SubgroupSize) {
        uint r=j/16, d=j%16;
        if(row+r<p.rows && column+d<p.cols) {
            uint index=(row+r)*p.cols+column+d;
            y[index]=tile_c[base+j]+(((p.flags&1)!=0)?z[index]:0.0);
        }
    }
}
#endif
void main() {
#ifdef COOPERATIVE
    if(p.op==1 && p.inner%16==0 && p.cols%16==0) { cooperative_linear(); return; }
#endif
    uint i=(gl_WorkGroupID.y*gl_NumWorkGroups.x+gl_WorkGroupID.x)*64+gl_LocalInvocationID.x;
    if(p.op==0) { // embedding gather
        if(i>=p.rows*p.cols) return;
        y[i]=float(w[uint(ids[i/p.cols])*p.cols+i%p.cols]);
    } else if(p.op==1) { // row-major linear projection; weights are [output,input]
        if(i>=p.rows*p.cols) return;
        uint row=i/p.cols, col=i%p.cols;
        float a=0.0;
        for(uint k=0;k<p.inner;k++) a+=float(float16_t(x[row*p.inner+k]))*float(w[(col+p.offset)*p.inner+k]);
        if((p.flags&1)!=0) a+=z[i];
        y[i]=a;
    } else if(p.op==2) { // LayerNorm or RMSNorm
        if(i>=p.rows) return;
        float mean=0.0, variance=0.0;
        for(uint k=0;k<p.cols;k++) mean+=x[i*p.cols+k];
        mean=(p.flags==0)?mean/float(p.cols):0.0;
        for(uint k=0;k<p.cols;k++) {float a=x[i*p.cols+k]-mean; variance+=a*a;}
        float inv=inversesqrt(variance/float(p.cols)+p.eps);
        for(uint k=0;k<p.cols;k++) y[i*p.cols+k]=(x[i*p.cols+k]-mean)*inv*float(w[k]);
    } else if(p.op==3) { // residual addition
        if(i<p.rows*p.cols) y[i]=x[i]+z[i];
    } else if(p.op==4) { // GELU-GLU (ModernBERT) or SwiGLU (Qwen3)
        if(i>=p.rows*p.cols) return;
        float a=x[(i/p.cols)*p.stride+i%p.cols];
        float b=(p.flags==0)?x[(i/p.cols)*p.stride+p.cols+i%p.cols]:z[i];
        y[i]=((p.flags==0)?gelu(a):a/(1.0+exp(-a)))*b;
    } else if(p.op==5) { // split-half RoPE, including a slice of fused QKV
        if(i>=p.rows*p.cols) return;
        uint row=i/p.cols, channel=i%p.cols, d=channel%p.dim;
        uint mate=(d<p.dim/2)?channel+p.dim/2:channel-p.dim/2;
        float angle=float(row%p.seq)*pow(p.theta,-float(d%(p.dim/2))*2.0/float(p.dim));
        y[i]=x[row*p.stride+p.offset+channel]*cos(angle)
            +x[row*p.stride+p.offset+mate]*sin(angle)*((d<p.dim/2)?-1.0:1.0);
    } else if(p.op==6) { // online softmax: no sequence-squared allocation
        if(i>=p.rows*p.heads) return;
        uint row=i/p.heads, head=i%p.heads, pos=row%p.seq, batch=row/p.seq;
        uint kh=head/(p.heads/p.kv_heads);
        float acc[256]; for(uint d=0;d<p.dim;d++) acc[d]=0.0;
        float maximum=-3.402823466e+38, denominator=0.0;
        for(uint key=0;key<p.seq;key++) {
            uint kr=batch*p.seq+key;
            if(mask[kr]==0 || ((p.flags&1)!=0 && key>pos)) continue;
            if(p.window!=0 && abs(int(key)-int(pos))>int(p.window)) continue;
            float score=0.0;
            for(uint d=0;d<p.dim;d++) score+=x[row*p.heads*p.dim+head*p.dim+d]*z[kr*p.kv_heads*p.dim+kh*p.dim+d];
            score*=inversesqrt(float(p.dim));
            float next=max(maximum,score), correction=exp(maximum-next), probability=exp(score-next);
            denominator=denominator*correction+probability;
            for(uint d=0;d<p.dim;d++) acc[d]=acc[d]*correction+probability*v[kr*p.stride+p.offset+kh*p.dim+d];
            maximum=next;
        }
        for(uint d=0;d<p.dim;d++) y[row*p.heads*p.dim+head*p.dim+d]=acc[d]/max(denominator,1e-30);
    } else if(p.op==7) { // manifest-defined pooling
        if(i>=p.rows*p.cols) return;
        uint batch=i/p.cols, channel=i%p.cols;
        float a=0.0; uint count=0;
        for(uint k=0;k<p.seq;k++) {
            if(mask[batch*p.seq+k]==0) continue;
            if(p.flags==0) { if(k==0) a=x[(batch*p.seq+k)*p.cols+channel]; }
            else if(p.flags==1) a+=x[(batch*p.seq+k)*p.cols+channel];
            else a=x[(batch*p.seq+k)*p.cols+channel];
            count++;
        }
        y[i]=(p.flags==1)?a/float(count):a;
    } else if(p.op==8) { // dense-head activation
        if(i<p.rows*p.cols) y[i]=gelu(x[i]);
    } else if(p.op==9) { // sigmoid, or yes/no-only softmax
        if(i>=p.rows) return;
        float logit=(p.flags==0)?x[i]:x[i*2]-x[i*2+1];
        y[i]=1.0/(1.0+exp(-logit));
    } else if(p.op==10) { // L2-normalized embedding
        if(i>=p.rows) return;
        float sum=0.0; for(uint k=0;k<p.cols;k++) sum+=x[i*p.cols+k]*x[i*p.cols+k];
        float inverse=inversesqrt(max(sum,1e-30));
        for(uint k=0;k<p.cols;k++) y[i*p.cols+k]=x[i*p.cols+k]*inverse;
    } else if(p.op==11) {
        if(i<p.rows) y[i]=x[i];
    } else if(p.op==12) {
        if(i<p.rows) y[i]=1.0/(1.0+exp(z[i]-x[i]));
    } else if(p.op==13) {
        if(i<p.rows) y[i]=1.0/(1.0+exp(-x[i]-float(w[0])));
    }
}
