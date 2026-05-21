using System;
using System.Collections.Generic;
using System.Text;

namespace YoloSharpOnnx.TestCommon
{
    public class Yolo26
    {
        public const string Bus = "1 bus, 4 person [0.93, 0.92, 0.9, 0.86, 0.53]";
        public const string Zidane = "2 person, 1 tie [0.92, 0.9, 0.53]";

        public const string Cls01 = "broccoli 0.42, mashed_potato 0.19, ice_cream 0.11, mixing_bowl 0.05, zucchini 0.02";
        public const string Cls02 = "banana 0.81, lemon 0.02, broccoli 0.02, pineapple 0.02, orange 0.02";

        public const string Seg01 = "1 cat, 1 car [0.91, 0.77]";
        public const string Seg02 = "2 person, 1 tie [0.92, 0.9, 0.59]";

        public const string Pose01 = "1 person [0.92]";
        public const string Pose02 = "2 person [0.91, 0.85]";
    }
}
