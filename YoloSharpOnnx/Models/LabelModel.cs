using System;
using System.Collections.Generic;
using System.Text;

namespace YoloSharpOnnx.Models
{
    public struct LabelModel
    {
        public int Index { get; set; }

        public string Name { get; set; }


        public LabelModel(int index, string name)
        {
            Index = index;
            Name = name;
        }
    }
}
