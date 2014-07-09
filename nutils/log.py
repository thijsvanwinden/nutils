# -*- coding: utf8 -*-
#
# Module LOG
#
# Part of Nutils: open source numerical utilities for Python. Jointly developed
# by HvZ Computational Engineering, TU/e Multiscale Engineering Fluid Dynamics,
# and others. More info at http://nutils.org <info@nutils.org>. (c) 2014

"""
The log module provides print methods ``debug``, ``info``, ``user``,
``warning``, and ``error``, in increasing order of priority. Output is sent to
stdout as well as to an html formatted log file if so configured.
"""

from __future__ import print_function
import sys, time, os, warnings, re
from . import core

warnings.showwarning = lambda message, category, filename, lineno, *args: \
  warning( '%s: %s\n  In %s:%d' % ( category.__name__, message, filename, lineno ) )

LEVELS = 'path', 'error', 'warning', 'user', 'info', 'progress', 'debug'


# references to objects that are going to be redefined
_range = range
_iter = iter
_enumerate = enumerate


## LOGGERS, STREAMS

class devnull( object ):
  @staticmethod
  def write( text ):
    pass

def SimpleLog( level, *contexts ): # just writes to stdout
  verbosity = core.getprop( 'verbosity', 6 )
  if level in LEVELS[ verbosity: ]:
    return devnull
  sys.stdout.writelines( '%s > ' % context for context in contexts )
  return sys.stdout

class HtmlLog( object ):
  'html log'

  def __init__( self, html ):
    self.html = html

  def __call__( self, level, *contexts ):
    return HtmlStream( level, contexts, self.html )

class HtmlStream( object ):
  'html line stream'

  def __init__( self, level, contexts, html ):
    'constructor'

    self.out = SimpleLog( level, *contexts )
    self.level = level
    self.head = ''.join( '%s &middot; ' % context for context in contexts )
    self.body = ''
    self.html = html

  def write( self, text ):
    'write to out and buffer for html'

    self.out.write( text )
    self.body += text.replace( '<', '&lt;' ).replace( '>', '&gt;' )

  @staticmethod
  def _path2href( match ):
    whitelist = ['.jpg','.png','.svg','.txt'] + core.getprop( 'plot_extensions', [] )
    filename = match.group(0)
    ext = match.group(1)
    return '<a href="%s">%s</a>' % (filename,filename) if ext not in whitelist \
      else '<a href="%s" name="%s" class="plot">%s</a>' % (filename,filename,filename)

  def __del__( self ):
    'postprocess buffer and write to html'

    body = self.body
    if self.level == 'path':
      body = re.sub( r'\b\w+([.]\w+)\b', self._path2href, body )
    if self.level:
      body = '<span class="%s">%s</span>' % ( self.level, body )
    line = '<span class="line">%s</span>' % ( self.head + body )

    self.html.write( line )
    self.html.flush()

class ContextLog( object ):
  'static text with parent'

  def __init__( self, title, parent=None ):
    self.title = title
    self.parent = parent or _getlog()

  def __call__( self, level, *contexts ):
    return self.parent( level, self.title, *contexts )

class IterLog( object ):
  'iterable context logger that updates progress info'

  def __init__( self, title, iterator, length=None, parent=None ):
    self.title = title
    self.parent = parent or _getlog()
    self.length = length
    self.iterator = iterator
    self.index = -1

    # clock
    self.dt = core.getprop( 'progress_interval', 1. )
    self.dtexp = core.getprop( 'progress_interval_scale', 2 )
    self.dtmax = core.getprop( 'progress_interval_max', 0 )
    self.tnext = time.time() + self.dt

  def mktitle( self ):
    self.tnext = time.time() + self.dt
    return '%s %d' % ( self.title, self.index ) if self.length is None \
      else '%s %d/%d (%d%%)' % ( self.title, self.index, self.length, (self.index-.5) * 100. / self.length )

  def __iter__( self ):
    self.index = 0
    return self

  def next( self ):
    if time.time() > self.tnext:
      if self.dtexp != 1:
        self.dt *= self.dtexp
        if self.dt > self.dtmax > 0:
          self.dt = self.dtmax
      self.parent( 'progress' ).write( self.mktitle() + '\n' )
    self.index += 1
    try:
      return self.iterator.next()
    except:
      self.index = -1
      raise

  def __call__( self, level, *contexts ):
    return self.parent( level, self.mktitle(), *contexts ) if self.index >= 0 \
      else self.parent( level, *contexts )

class CaptureLog( object ):
  'capture output without printing'

  def __init__( self ):
    self.buf = ''

  def __nonzero__( self ):
    return bool( self.buf )

  def __str__( self ):
    return self.buf

  def __call__( self, level, *contexts ):
    for context in contexts:
      self.buf += '%s > ' % context
    return self

  def write( self, text ):
    self.buf += text

def _getlog():
  return core.getprop( 'log', SimpleLog )

def _getstream( level ):
  return _getlog()( level )

def _mklog( level ):
  return lambda *args, **kw: print( *args, file=_getstream(level), **kw )


## MODULE METHODS


locals().update({ level: _mklog(level) for level in LEVELS })

def range( title, *args ):
  items = _range( *args )
  return IterLog( title, _iter(items), len(items) )

def iter( title, iterable, length=None, parent=None ):
  return IterLog( title, _iter(iterable), len(iterable) if length is None else length, parent )

def enumerate( title, iterable, length=None, parent=None ):
  return IterLog( title, _enumerate(iterable), len(iterable) if length is None else length, parent )

def count( title, start=0, parent=None ):
  from itertools import count
  return IterLog( title, count(start), None, parent )
    
def stack( msg ):
  'print stack trace'

  from . import debug
  if isinstance( msg, tuple ):
    exc_type, exc_value, tb = msg
    msg = repr( exc_value )
    frames = debug.frames_from_traceback( tb )
  else:
    frames = debug.frames_from_callstack( depth=2 )
  print( msg, *reversed(frames), sep='\n', file=_getstream( 'error' ) )

def title( f ): # decorator
  def wrapped( *args, **kwargs ):
    __log__ = ContextLog( kwargs.pop( 'title', f.func_name ) )
    return f( *args, **kwargs )
  return wrapped


# vim:shiftwidth=2:foldmethod=indent:foldnestmax=2
