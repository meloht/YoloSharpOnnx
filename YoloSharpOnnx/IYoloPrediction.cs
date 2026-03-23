using System;
using System.Collections.Generic;
using System.Text;

namespace YoloSharpOnnx
{
    public interface IYoloPrediction<T>
    {
        internal abstract static string Describe(T[] predictResults);
    }
}
