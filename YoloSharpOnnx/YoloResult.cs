using System;
using System.Collections.Generic;
using System.Text;

namespace YoloSharpOnnx
{
    public class YoloResult<T>(T[] items) where T : IYoloPrediction<T>
    {
        public T[] Items { get; } = items;

        public SpeedResult SpeedResult { get; set; }

        public override string ToString() => T.Describe(Items);
    }
}
