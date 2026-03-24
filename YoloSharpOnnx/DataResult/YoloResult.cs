using System;
using System.Collections.Generic;
using System.Text;

namespace YoloSharpOnnx.DataResult
{
    public class YoloResult<T>(List<T> items, SpeedResult speed) where T : IYoloPrediction<T>
    {
        public List<T> Items { get; } = items;

        public SpeedResult SpeedResult { get; } = speed;

        public override string ToString() => T.Describe(Items);
    }
}
