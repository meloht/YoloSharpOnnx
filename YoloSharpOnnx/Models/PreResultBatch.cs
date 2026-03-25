using System;
using System.Collections.Generic;
using System.Text;

namespace YoloSharpOnnx.Models
{
    public record PreResultBatch(PreResult PreResult, string ImagePath, float[] Data);
   
}
