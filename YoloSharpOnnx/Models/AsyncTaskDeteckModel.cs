using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using YoloSharpOnnx.DataResult;

namespace YoloSharpOnnx.Models
{
    public struct AsyncTaskDeteckModel
    {
        public string ImagePath { get; set; }
        public Task<List<DetectionResult>> DetectionResults { get; set; }

        public Guid Id { get; set; }
    }
}
