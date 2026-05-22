using Microsoft.ML.OnnxRuntime;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using YoloSharpOnnx.DataResult;
using YoloSharpOnnx.Inference.Detect.Models;

namespace YoloSharpOnnx.Inference.Segment
{
    internal interface ISegPostprocess : IDisposable
    {
        List<SegResult> PostProcessSync(OrtValue outputValue0, OrtValue outputValue1, PreDetectResult preResult);
        List<SegResult> PostProcessAsync(OrtValue outputValue0, OrtValue outputValue1, PreDetectResult preResult);
    }
}
