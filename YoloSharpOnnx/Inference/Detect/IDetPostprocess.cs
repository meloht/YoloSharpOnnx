using Microsoft.ML.OnnxRuntime;
using System;
using System.Collections.Generic;
using System.Text;
using YoloSharpOnnx.DataResult;
using YoloSharpOnnx.Inference.Detect.Models;

namespace YoloSharpOnnx.Inference.Detect
{
    public interface IDetPostprocess : IDisposable
    {
        List<DetectionResult> PostProcessSync(OrtValue outputValue, PreDetectResult preResult);
        List<DetectionResult> PostProcessAsync(OrtValue outputValue, PreDetectResult preResult);
    }
}
