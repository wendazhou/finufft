#ifndef QUTE_H
#define QUTE_H

#include <string>
#include <vector>
#include <map>

class QuteString {
public:
    QuteString() {
    }
    QuteString(const QuteString &other) {
        m_data=other.m_data;
    }
    QuteString(std::string str) {
        m_data=str;
    }
    QuteString(const char *str) {
        m_data=str;
    }
    bool operator==(const QuteString &other) const {
        return (m_data==other.m_data);
    }
    char &operator[](int index);
    char value(int index) const;
    int count() const;
    QuteString mid(int start,int len=-1) const;
    bool startsWith(const QuteString &substr) const;
    int indexOf(const QuteString &substr) const;
    bool lessThan(const QuteString &other) const;
    float toFloat() const;
    double toDouble() const;
    int toInt() const;
    void append(const QuteString &str);

private:
    std::string m_data;
};

QuteString operator+(const QuteString &str1,const QuteString &str2);
bool operator<(const QuteString &str1,const QuteString &str2);

class QuteStringMap {
public:
    QuteStringMap() {}
    QuteStringMap(const QuteStringMap &other) {
        m_data=other.m_data;
    }
    QuteString &operator[](const QuteString &key);
    QuteString value(const QuteString &key,const QuteString &default_val="");
    bool contains(const QuteString &key) const;
private:
    std::map<QuteString,QuteString> m_data;
};

class QuteStringList {
public:
    QuteStringList() {}
    QuteStringList(const QuteStringList &other) {
        m_data=other.m_data;
    }
    int count() const;
    QuteString &operator[](int index);
    QuteString value(int index,const QuteString &default_val="");
    //void operator<<(const QuteString &str);
private:
    std::vector<QuteString> m_data;
};

#endif // QUTE_H

