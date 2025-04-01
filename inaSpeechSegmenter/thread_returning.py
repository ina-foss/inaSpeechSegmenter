#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 27 15:18:49 2018

@author: elechapt

This module provides an extension to Python's threading capabilities by implementing a custom Thread class,
ThreadReturning, that allows threads to return a value when they complete. By overriding the run and join 
methods, ThreadReturning captures the result of the target function executed in the thread, making it easier 
to retrieve the output after the thread finishes execution.
"""


from threading import Thread

class ThreadReturning(Thread):
    """
    Allow us to get the results from a thread
    """
    def __init__(self, *args, **kwargs):
        Thread.__init__(self, *args, **kwargs)
        self._return = None

    def run(self):
        if self._target is not None:
            self._return = self._target(*self._args, **self._kwargs)

    def join(self):
        Thread.join(self)
        return self._return


