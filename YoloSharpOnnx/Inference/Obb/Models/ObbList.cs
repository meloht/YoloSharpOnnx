using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using YoloSharpOnnx.DataResult;

namespace YoloSharpOnnx.Inference.Obb.Models
{
    internal class ObbList : IDisposable
    {
        public List<ObbResult> Results { get; } = new List<ObbResult>();

        public void Dispose()
        {
            Results?.Clear();
        }
    }
}
