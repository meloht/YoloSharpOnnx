using Microsoft.ML.OnnxRuntime;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using YoloSharpOnnx.DataResult;
using YoloSharpOnnx.Inference.Detect.Models;

namespace YoloSharpOnnx.Inference.DetectCore
{
    internal interface IDetCorePostprocess<TResult> : IDisposable
    {
        List<TResult> PostProcessSync(OrtValue outputValue, PreDetectResult preResult);
        List<TResult> PostProcessAsync(OrtValue outputValue, PreDetectResult preResult);
    }
}
