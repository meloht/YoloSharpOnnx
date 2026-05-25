using Microsoft.ML.OnnxRuntime;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using YoloSharpOnnx.DataResult;
using YoloSharpOnnx.Inference.Detect.Models;

namespace YoloSharpOnnx.Inference.Obb
{
    internal interface IObbPostprocess : IDisposable
    {
        List<ObbResult> PostProcessSync(OrtValue output, PreDetectResult preResult);
        List<ObbResult> PostProcessAsync(OrtValue output, PreDetectResult preResult);
    }
}
