# SimUESR 
Simultaneously enhancing the visibility and super-resolution of underwater images poses a challenging task that
demands a concerted effort to overcome. In spite of the emergence
of various deep learning models, almost all existing methods are
tailored to specific enhancement tasks, rendering them unsuitable
for underwater scenes. Consequently, most methods result in
color distortion, blurred structures, and missing high-frequency
details. To address this challenge, this paper proposes a multi-level degradation removal enhancer that utilizes underwater
transmission prior to simultaneously perform joint enhancement
and super-resolution tasks, named as SimUESR. Specifically, the
proposed SimUESR is designed to be guided by multiple sets of
transmission-inspired guidance, which are cascaded with multi-stage degradation removal modules via a feature modulation
operation. Through this, the underwater prior is used as modulation information to modulate contrast and color deviation,
gradually embedding it in the transmission-guided modules at the
feature level. Then the enhanced features are incorporated into
a multi-stage degradation removal module to generate lossless
image content. To release the burden of manually designing loss,
we introduce a novel bilevel adaptive learning adaption strategy
that combines finite-difference approximation to automatically
search for the desired loss, effectively improving visual perception
performance.

## Pipline 
 **Network architecture and learning strategy**
![image]([https://github.com/lpm1001/SimUESR/blob/main/resources/pipeline.pdf](https://github.com/lpm1001/SimUESR/blob/main/resources/pipeline.png))  
## Experiment results

 **Qualitative comparison for super-resolution performance on the USR-248 dataset**
![image](https://github.com/lpm1001/SimUESR/blob/main/resources/usrx2.png)
![image](https://github.com/lpm1001/SimUESR/blob/main/resources/usrx4.png)
 **Qualitative comparison for super-resolution performance on the UFO-120 dataset**
![image](https://github.com/lpm1001/SimUESR/blob/main/resources/ufox2.png)
![image](https://github.com/lpm1001/SimUESR/blob/main/resources/ufox4.png)
## Contact: 
If you have any question, please email lipeiming1001@163.com
## Citation:    
```
@Override
protected void onDestroy() {
    EventBus.getDefault().unregister(this);
    super.onDestroy();
}
```  
