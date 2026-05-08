using System;
using System.Collections.Generic;
using System.Text;
using YoloSharpOnnx.Inference.Detect;

namespace YoloSharpOnnx
{
    public interface IExecutionProvider
    {
        IYoloDetect CreateYoloDetect();
        void SetYoloConfiguration(YoloConfig yoloConfig);
    }
}
