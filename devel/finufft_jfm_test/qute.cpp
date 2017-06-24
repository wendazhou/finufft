#include "qute.h"

QuteString QuteString::mid(int start, int len) const
{
    std::string str=m_data;
    if (start>=(int)str.size()) return "";
    if (len<0) len=str.size()-start;
    if (start+len>(int)str.size()) len=str.size()-start;
    std::string ret;
    for (int i=start; i<start+len; i++) {
        ret.push_back(str.at(i));
    }
    return ret;
}

bool QuteString::startsWith(const QuteString &substr) const
{
    std::string str=m_data;
    if ((int)str.size()<substr.count()) return false;
    for (int i=0; i<substr.count(); i++) {
        if (str.at(i)!=substr.value(i)) return false;
    }
    return true;
}

int QuteString::indexOf(const QuteString &substr) const
{
    for (int i=0; i<this->count(); i++) {
        if (this->mid(i).startsWith(substr))
            return i;
    }
    return -1;
}

bool QuteString::lessThan(const QuteString &other) const
{
    return (m_data<other.m_data);
}

float QuteString::toFloat() const
{
    return atof(m_data.data());
}

double QuteString::toDouble() const
{
    return atof(m_data.data());
}

int QuteString::toInt() const
{
    return atoi(m_data.data());
}

void QuteString::append(const QuteString &str)
{
    for (int i=0; i<str.count(); i++) {
        m_data.push_back(str.value(i));
    }
}

char &QuteString::operator[](int index)
{
    return m_data.at(index);
}

char QuteString::value(int index) const
{
    if ((index<0)||(index>=count())) return 0;
    return m_data.at(index);
}

int QuteString::count() const
{
    return m_data.size();
}

QuteString &QuteStringMap::operator[](const QuteString &key)
{
    return m_data[key];
}

QuteString QuteStringMap::value(const QuteString &key, const QuteString &default_val)
{
    if (m_data.find(key)==m_data.end())
        return default_val;
    return m_data[key];
}

bool QuteStringMap::contains(const QuteString &key) const
{
    return (m_data.find(key)!=m_data.end());
}

int QuteStringList::count() const
{
    return m_data.size();
}

QuteString &QuteStringList::operator[](int index)
{
    return m_data[index];
}

QuteString QuteStringList::value(int index, const QuteString &default_val)
{
    if ((index<0)||(index>=(int)m_data.size()))
        return default_val;
    return m_data[index];
}

/*void QuteStringList::operator<<(const QuteString &str)
{
    m_data.push_back(str);
}*/


QuteString operator+(const QuteString &str1, const QuteString &str2)
{
    QuteString ret=str1;
    ret.append(str2);
    return ret;
}

bool operator<(const QuteString &str1, const QuteString &str2)
{
    return str1.lessThan(str2);
}
