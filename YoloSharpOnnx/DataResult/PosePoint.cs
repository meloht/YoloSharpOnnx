using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace YoloSharpOnnx.DataResult
{

    public record PosePoint(float X, float Y,int Index, float Confidence);
}
