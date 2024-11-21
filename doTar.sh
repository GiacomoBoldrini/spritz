#! /bin/bash
tar --exclude=".git" --exclude=".gitignore" --exclude="data/common/__pycache__" -zcvf spritz.tar.xz *
