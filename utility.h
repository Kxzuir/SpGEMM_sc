#pragma once
#ifndef UTILITY_H
#define UTILITY

unsigned int upper2bound(unsigned int v);
void printHeader(const char *appName, int majorVer, int minorVer, int year, const char *corpName);
void printHelp(char *appName, char *opts, char *info);
void printLine(int lineLen = 80);
void clearLine(int charCnt = 80);
#endif // !UTILITY_H