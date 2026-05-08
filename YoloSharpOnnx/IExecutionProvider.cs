using System;
using System.Collections.Generic;
using System.Text;
using YoloSharpOnnx.Inference.Classify;
using YoloSharpOnnx.Inference.Detect;

namespace YoloSharpOnnx
{
    public interface IExecutionProvider
    {
        IYoloDetect CreateYoloDetect();
        IYoloClassify CreateYoloClassify();
        void SetYoloConfiguration(YoloConfig yoloConfig);
    }
}
