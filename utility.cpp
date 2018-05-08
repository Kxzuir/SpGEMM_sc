#include "utility.h"
#include <cstdio>
#include <cstdarg>
#include <cstring>
using namespace std;

unsigned int upper2bound(unsigned int v)
{
	v--;
	v |= v >> 1;
	v |= v >> 2;
	v |= v >> 4;
	v |= v >> 8;
	v |= v >> 16;
	v++;
	return v;
}

void printHeader(const char *appName, int majorVer, int minorVer, int year, const char *corpName)
{
	printf("%s [Version %d.%d]\n", appName, majorVer, minorVer);
	printf("(c) %d %s. All rights reserved.\n", year, corpName);
	printf("\n");
}

/*
Usage:
mkfs [options] [-t <type>] [fs-options] <device> [<size>]

Make a Linux filesystem.

*/

void printHelp(char *appName, char *opts, char *info)
{
	printf("Usage:\n");
	printf("%s %s\n\n", appName, opts);
	printf("%s", info);
}

void printLine(int lineLen)
{
	for (int i = 0; i < lineLen; i++) printf("-");
	printf("\n");
}

void clearLine(int charCnt)
{
	for (int i = 0; i < charCnt; i++) printf("\b");
	for (int i = 0; i < charCnt; i++) printf(" ");
	for (int i = 0; i < charCnt; i++) printf("\b");
}