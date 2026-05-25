using System;
using System.Collections.Generic;
using System.Text;

namespace YoloSharpOnnx.TestCommon
{
    public class Yolo8
    {
        public const string Bus = "1 bus, 4 person [0.84, 0.89, 0.88, 0.88, 0.44]";
        public const string Zidane = "2 person [0.83, 0.83]";

        public const string Cls01 = "mashed_potato 0.3, broccoli 0.21, mixing_bowl 0.1, meat_loaf 0.08, cucumber 0.07";
        public const string Cls02 = "banana 0.88, pomegranate 0.02, lemon 0.01, orange 0.01, spaghetti_squash 0.01";

        public const string Seg01 = "1 cat, 2 car [0.85, 0.54, 0.46]";
        public const string Seg02 = "2 person, 1 tie [0.86, 0.85, 0.66]";

        public const string Pose01 = "1 person [0.86]";
        public const string Pose02 = "2 person [0.9, 0.88]";

        public const string Obb01 = "6 plane [0.92, 0.91, 0.9, 0.86, 0.72, 0.71]";
        public const string Obb02 = "3 plane [0.94, 0.93, 0.91]";
    }
}
