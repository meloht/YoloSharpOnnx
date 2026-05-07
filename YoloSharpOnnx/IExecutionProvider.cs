using System;
using System.Collections.Generic;
using System.Text;

namespace YoloSharpOnnx
{
    public interface IExecutionProvider
    {
        IYoloDetect CreateYoloDetect();
        void SetYoloConfiguration(YoloConfig yoloConfig);
    }
}
