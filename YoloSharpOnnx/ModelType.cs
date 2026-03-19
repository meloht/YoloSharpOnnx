using System;
using System.Collections.Generic;
using System.Runtime.Serialization;
using System.Text;

namespace YoloSharpOnnx
{
    public enum ModelType
    {
        [EnumMember(Value = "classify")]
        Classification,

        [EnumMember(Value = "detect")]
        ObjectDetection,

        [EnumMember(Value = "obb")]
        ObbDetection,

        [EnumMember(Value = "segment")]
        Segmentation,

        [EnumMember(Value = "pose")]
        PoseEstimation
    }
}
