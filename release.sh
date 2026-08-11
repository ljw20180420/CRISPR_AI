#!/bin/bash

# change to the dir of the script
cd $( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

bump() {
    local major
    local minor
    local patch

    IFS="." read major minor patch < .version
    patch=$(( patch + 1 ))
    printf "%d.%d.%d\n" ${major} ${minor} ${patch}
}

version=$(bump)
echo ${version} > .version
git commit -am "release ${version}"
git push
gh release create "${version}" \
    --generate-notes \
    --fail-on-no-commits
git fetch --tags origin
