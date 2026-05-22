using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace YoloSharpOnnx.Inference
{
    internal interface IGuidValue<T>
    {
        public Guid Guid { get; set; }

        public T PreResult { get; set; }
    }
}
