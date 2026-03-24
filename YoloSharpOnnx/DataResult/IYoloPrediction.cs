using System;
using System.Collections.Generic;
using System.Text;

namespace YoloSharpOnnx.DataResult
{
    public interface IYoloPrediction<T>
    {
        internal abstract static string Describe(List<T> predictResults);
    }
}
