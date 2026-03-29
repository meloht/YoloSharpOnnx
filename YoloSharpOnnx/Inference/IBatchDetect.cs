using System;
using System.Collections.Generic;
using System.Text;
using YoloSharpOnnx.DataResult;
using YoloSharpOnnx.Models;

namespace YoloSharpOnnx.Inference
{
    public interface IBatchDetect
    {
        List<DetectionResult> RunBatchDetect(PreResultBatch preRes, YoloConfig yoloConfig);
    }
}
