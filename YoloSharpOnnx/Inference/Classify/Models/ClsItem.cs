using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace YoloSharpOnnx.Inference.Classify.Models
{
    public struct ClsItem
    {
        public int Index; 
        public float Value;
        public ClsItem(int index, float value)
        {
            Index = index;
            Value = value;
        }
    }
}
