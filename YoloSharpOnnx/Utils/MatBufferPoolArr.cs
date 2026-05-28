using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using YoloSharpOnnx.Models;

namespace YoloSharpOnnx.Utils
{
    public sealed class MatBufferPoolArr
    {
        /// <summary>
        /// 实际缓存池
        /// </summary>
        private readonly ImageBatchData[] _pool;

        /// <summary>
        /// 最大缓存数量（超过则直接 Dispose）
        /// </summary>
        private readonly int _maxSize;


        /// <summary>
        /// 当前正在使用中的对象数量
        /// </summary>
        private int _usedCount;

        /// <summary>
        /// 用于创建对象
        /// </summary>
        private readonly OnnxModel _onnxModel;

        private bool _disposed;
        private readonly object _locker = new object();
        private int _currentIndex = 0;

        public MatBufferPoolArr(int maxSize, OnnxModel onnxModel)
        {
            ArgumentOutOfRangeException.ThrowIfNegativeOrZero(maxSize);
            _maxSize = maxSize;
            _onnxModel = onnxModel ?? throw new ArgumentNullException(nameof(onnxModel));
            _pool = new ImageBatchData[maxSize];
            // 预热池
            for (int i = 0; i < _maxSize; i++)
            {
                _pool[i] = new ImageBatchData(_onnxModel);
            }
            _currentIndex = _pool.Length - 1; 
        }

        /// <summary>
        /// 当前使用中的对象数量
        /// </summary>
        public int UsedCount => Volatile.Read(ref _usedCount);



        public ImageBatchData Rent()
        {
            ThrowIfDisposed();

            lock (_locker)
            {
                _usedCount++;
                if (_pool[_currentIndex] != null)
                {
                    var res = _pool[_currentIndex];
                    _pool[_currentIndex] = null;
                    _currentIndex--;
                    if (_currentIndex < 0)
                    {
                        _currentIndex = 0;
                    }
                   
                    return res;
                }
                else
                {
                    // 池空了，临时创建
                    return new ImageBatchData(_onnxModel);
                }
               
            }

        }

        private void Test()
        {
            for (int i = 0; i < _pool.Length; i++)
            {
                if (_pool[i] != null)
                {
                    Console.Write("1, ");
                }
                else
                {
                    Console.Write("0, ");
                }
            }
            Console.WriteLine();
        }
        /// <summary>
        /// 归还对象
        /// </summary>
        public void Return(ImageBatchData item)
        {
            if (item == null)
                return;

            if (_disposed)
            {
                item.Dispose();
                return;
            }

            lock (_locker)
            {
                _usedCount--;
                if (_pool[_currentIndex] != null)
                {
                    if (_currentIndex != _pool.Length - 1)
                    {
                        _currentIndex++;
                        _pool[_currentIndex] = item;

                    }
                    else
                    {
                        item.Dispose();
                    }
                }
                else
                {
                    _pool[_currentIndex] = item;
                }

            }
              
        }

        /// <summary>
        /// 清空池
        /// </summary>
        public void Clear()
        {
            for (int i = 0; i < _pool.Length; i++)
            {
                _pool[i]?.Dispose();
                _pool[i] = null;
            }
            _usedCount = 0;
        }
        private void ThrowIfDisposed()
        {
            ObjectDisposedException.ThrowIf(_disposed, nameof(MatBufferPoolArr));
        }
        public void Dispose()
        {
            if (_disposed)
                return;

            _disposed = true;

            Clear();

            GC.SuppressFinalize(this);
        }
    }
}
