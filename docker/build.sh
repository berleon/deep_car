#! /usr/bin/env bash

cp ~/.ssh/id_rsa.pub authorized_keys
docker build -t deep_car .
rm authorized_keys



