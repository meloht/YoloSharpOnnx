using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Linq;
using System.Reflection;
using System.Runtime.Serialization;
using System.Text;
using System.Threading.Tasks;

namespace YoloSharpOnnx
{
    public static class EnumExtensions
    {
        private static readonly ConcurrentDictionary<YoloTaskType, string> _cache = new();
        public static string GetDescription(this YoloTaskType value)
        {
            if(_cache.ContainsKey(value))
                return _cache[value];
            FieldInfo field = value.GetType().GetField(value.ToString());

            if (field == null)
            {
                AddEnumCache(value, value.ToString());
                return value.ToString();
            }

            EnumMemberAttribute attribute = field.GetCustomAttribute<EnumMemberAttribute>();

            string val = attribute?.Value ?? value.ToString();
            AddEnumCache(value, val);

            return val;
        }

        private static void AddEnumCache(YoloTaskType enumType, string val)
        {
            if (string.IsNullOrEmpty(val))
                return;
            if (!_cache.ContainsKey(enumType))
            {
                _cache.TryAdd(enumType, val);
            }
        }
    }
}
