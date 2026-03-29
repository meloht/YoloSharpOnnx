using Microsoft.ML.OnnxRuntime;
using System;
using System.Collections.Generic;
using System.Text;
using YoloSharpOnnx.DataResult;
using YoloSharpOnnx.Models;

namespace YoloSharpOnnx.Inference
{
    public interface IPostprocess
    {
        public List<DetectionResult> PostProcess(OrtValue outputValue, PreResult preResult, YoloConfig yoloConfig);
    }
}
