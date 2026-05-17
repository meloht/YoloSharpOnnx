using Microsoft.ML.OnnxRuntime;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using YoloSharpOnnx.Inference.Detect.Models;

namespace YoloSharpOnnx.Inference.Segment.Models
{
    public record InferModel<T>(OrtValue Output0, OrtValue Output1, T TBatchPreResult,long StartTime);

}
