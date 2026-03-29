using OpenCvSharp;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using YoloSharpOnnx.Models;

namespace YoloSharpOnnx.Inference
{
    public class MatBufferPool : IDisposable
    {
        private ImageBatchData[] _matPool;
        private int _valIdx = 0;
        private int _nullIdx = -1;
        private readonly object _lock = new object();

        private readonly int _poolSzie;
        private OnnxModel _OnnxModel;
        private readonly bool[] _flagArr;

        public MatBufferPool(int poolSzie, OnnxModel onnxModel)
        {
            _poolSzie = poolSzie;
            _OnnxModel = onnxModel;
            _flagArr = new bool[_poolSzie];

            _matPool = new ImageBatchData[_poolSzie];
            for (int i = 0; i < _poolSzie; i++)
            {
                _matPool[i] = new ImageBatchData(_OnnxModel);
            }

        }


        public ImageBatchData Rent()
        {
            lock (_lock)
            {
                if (_valIdx < _matPool.Length && _valIdx >= 0 && _flagArr[_valIdx])
                {
                    var mat = _matPool[_valIdx];
                    _flagArr[_valIdx] = false;

                    _nullIdx = _valIdx;
                    _valIdx++;

                    if (_valIdx > _matPool.Length - 1)
                    {
                        _valIdx = _matPool.Length - 1;
                    }
                    // Test();
                    return mat;
                }
                else
                {
                    //Test();
                    return new ImageBatchData(_OnnxModel);
                }

            }

        }
        public void Return(ImageBatchData mat)
        {
            lock (_lock)
            {
                if (_nullIdx < _matPool.Length && _nullIdx >= 0)
                {
                    _matPool[_nullIdx] = mat;
                    _flagArr[_nullIdx] = true;

                    _valIdx = _nullIdx;
                    _nullIdx--;

                    if (_nullIdx < 0)
                    {
                        _nullIdx = 0;
                    }
                }
                else
                {
                    mat.Dispose();
                }
                // Test();
            }



        }

        //public void Test()
        //{
        //    StringBuilder sb = new StringBuilder();
        //    for (int i = 0; i < _matPool.Length; i++)
        //    {
        //        if (_matPool[i] == null)
        //        {
        //            sb.Append("null, ");
        //        }
        //        else
        //        {
        //            sb.Append("value, ");
        //        }

        //    }
        //    Console.WriteLine(sb.ToString());
        //}

        public void Dispose()
        {
            for (int i = 0; i < _matPool.Length; i++)
            {
                if (_matPool[i] != null)
                {
                    _matPool[i].Dispose();
                    _matPool[i] = null;
                }

            }
            Array.Clear(_matPool);
            _matPool = null;
        }
    }
}
