using System;
using System.Collections.Generic;
using System.Text;
using YoloSharpOnnx.DataResult;
using YoloSharpOnnx.Inference.Detect.Models;

namespace YoloSharpOnnx.Inference.Detect
{
    public interface IBatchDetect
    {
        List<DetectionResult> RunBatchDetect(PreDetectResultBatch preRes);
    }
}
