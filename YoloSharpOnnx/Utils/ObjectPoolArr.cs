using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using YoloSharpOnnx.Models;

namespace YoloSharpOnnx.Utils
{
    public sealed class ObjectPoolArr<T> : IDisposable where T : class, IDisposable
    {
        /// <summary>
        /// 创建对象的方法
        /// </summary>
        private readonly Func<T> _factory;
        private readonly Action<T>? _resetAction;

        /// <summary>
        /// 最大缓存数量
        /// 防止无限增长
        /// </summary>
        private readonly int _maxSize;

        /// <summary>
        /// 使用 ConcurrentBag 保证高性能线程安全
        /// </summary>
        private readonly T[] _items;


        private bool _disposed;
        private readonly object _locker = new object();
        private int _currentIndex = 0;

        public ObjectPoolArr(Func<T> factory, int maxSize = 30, Action<T>? resetAction = null)
        {
            _factory = factory ?? throw new ArgumentNullException(nameof(factory));
            _maxSize = maxSize > 0 ? maxSize : 30;
            _resetAction = resetAction;

            _items = new T[_maxSize];
           
        }

        private void ThrowIfDisposed()
        {
            ObjectDisposedException.ThrowIf(_disposed, instance: this);
        }
        public void Dispose()
        {
            if (_disposed)
                return;

            _disposed = true;

            Clear();

            GC.SuppressFinalize(this);
        }

        /// <summary>
        /// 获取对象
        /// </summary>
        public T Rent()
        {
            ThrowIfDisposed();
            lock (_locker)
            {
                if (_items[_currentIndex] != null)
                {
                    var res = _items[_currentIndex];
                    _items[_currentIndex] = null;
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
                    return _factory();
                }

            }

         
        }
        /// <summary>
        /// 清空对象池
        /// </summary>
        private void Clear()
        {
            for (int i = 0; i < _items.Length; i++)
            {
                _items[i]?.Dispose();
                _items[i] = null;
            }
        }
        /// <summary>
        /// 归还对象
        /// </summary>
        public void Return(T item)
        {
            if (item == null)
                return;

            if (_disposed)
            {
                item?.Dispose();
                return;
            }
            _resetAction?.Invoke(item);
            // 超过最大容量则直接销毁
            lock (_locker)
            {

                if (_items[_currentIndex] != null)
                {
                    if (_currentIndex != _items.Length - 1)
                    {
                        _currentIndex++;
                        _items[_currentIndex] = item;

                    }
                    else
                    {
                        item.Dispose();
                    }
                }
                else
                {
                    _items[_currentIndex] = item;
                }

            }
        }
    }
}
