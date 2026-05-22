using System;
using System.Collections.Generic;
using System.Text;

namespace YoloSharpOnnx.TestCommon
{
    public class Yolo11
    {
        public const string Bus = "1 bus, 4 person [0.94, 0.9, 0.85, 0.83, 0.4]";
        public const string Zidane = "2 person, 1 tie [0.86, 0.79, 0.48]";

        public const string Cls01 = "broccoli 0.41, mashed_potato 0.13, fig 0.07, ice_cream 0.04, meat_loaf 0.04";
        public const string Cls02 = "banana 0.22, lemon 0.15, spaghetti_squash 0.09, orange 0.07, pineapple 0.06";

        public const string Seg01 = "1 cat, 1 car [0.91, 0.64]";
        public const string Seg02 = "2 person, 1 tie [0.9, 0.8, 0.33]";

        public const string Pose01 = "1 person [0.9]";
        public const string Pose02 = "2 person [0.9, 0.85]";

        public const string Obb01 = "1 person [0.9]";
        public const string Obb02 = "2 person [0.9, 0.85]";
    }
}
