using OpenCvSharp;
using System;
using System.Collections.Generic;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using YoloSharpOnnx.DataResult;
using YoloSharpOnnx.Models;

namespace YoloSharpOnnx
{
    public class HumanSkeleton : ISkeleton
    {
        private readonly string[] _colors =
        [
            "#FF8000",
            "#FF9933",
            "#FFB266",
            "#E6E600",
            "#FF99FF",
            "#99CCFF",
            "#FF66FF",
            "#FF33FF",
            "#66B2FF",
            "#3399FF",
            "#FF9999",
            "#FF6666",
            "#FF3333",
            "#99FF99",
            "#66FF66",
            "#33FF33",
            "#00FF00",
            "#0000FF",
            "#FF0000",
            "#FFFFFF",
        ];

        private readonly (int, int)[] _skeletonConnections =
        [
            (16, 14),
            (14, 12),
            (17, 15),
            (15, 13),
            (12, 13),
            (6, 12),
            (7, 13),
            (6, 7),
            (6, 8),
            (7, 9),
            (8, 10),
            (9, 11),
            (2, 3),
            (1, 2),
            (1, 3),
            (2, 4),
            (3, 5),
            (4, 6),
            (5, 7),
        ];

        private readonly int[] _keypointColorMap =
        [
            16, 16, 16, 16, 16, 0, 0, 0, 0, 0, 0, 9, 9, 9, 9, 9, 9
        ];

        private readonly int[] _lineColorMap =
        [
            9, 9, 9, 9, 7, 7, 7, 0, 0, 0, 0, 0, 16, 16, 16, 16, 16, 16, 16
        ];

        public int ConnectionCount => _skeletonConnections.Length;

        public Scalar GetKeypointColor(int index)
        {
            index = _keypointColorMap[index % _keypointColorMap.Length];

            var hex = _colors[index];

            return ColorTemplate.HexToRgbaScalar(hex);
        }

        public Scalar GetLineColor(int index)
        {
            index = _lineColorMap[index % _lineColorMap.Length];

            var hex = _colors[index];

            return ColorTemplate.HexToRgbaScalar(hex);
        }

        public PosePoint GetKeypoint1(int index, PosePoint[] keyPoints)
        {
            return keyPoints[_skeletonConnections[index].Item1 - 1];
        }
        public PosePoint GetKeypoint2(int index, PosePoint[] keyPoints)
        {
            return keyPoints[_skeletonConnections[index].Item2 - 1];
        }
    }

    
}
