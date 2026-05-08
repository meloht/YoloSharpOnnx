using System;
using System.Collections.Generic;
using System.Text;
using YoloSharpOnnx.DataResult;
using YoloSharpOnnx.Models;

namespace YoloSharpOnnx.Inference.Detect
{
    public interface IBatchDetect
    {
        List<DetectionResult> RunBatchDetect(PreResultBatch preRes);
    }
}
