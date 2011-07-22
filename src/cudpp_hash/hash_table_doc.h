/*!
\mainpage CUDA Hash Table Documentation
\version Unofficial release, 0.75

\section Introduction
This library allows you to build and use hash tables directly on a GPU.

<!--
\section Homepage
-->

\section Information
Three different hash table types are included:
<table style="width:90%; margin: auto; border: 1px solid #dddddd;">
  <tr>
    <th class="classes" style="padding-right: 1em;">
      \ref CudaHT::CuckooHashing::HashTable
    </th>
    <td>
      Stores a single value per key.  Input is expected to be a set of key-value pairs, where the keys are all unique.
    </td>
  </tr>
  <tr>
    <th class="classes" style="padding-right: 1em;">
      \ref CudaHT::CuckooHashing::CompactingHashTable
    </th>
    <td>
      Assigns each key a unique identifier and allows O(1) translation between the key and the unique IDs.
      Input is a set of keys that may, or may not, be repeated.
    </td>
  </tr>
  <tr>
    <th class="classes" style="padding-right: 1em;">
      \ref CudaHT::CuckooHashing::MultivalueHashTable
    </th>
    <td>
      Allows you to store multiple values for ach key.
      Multiple values for the same key are represented by different key-value pairs in the input.
    </td>
  </tr>
</table>

\section Building Building the library

\subsection Libraries Required libraries
This library relies on a good random number generator to ensure good hash function generation.
We use the Mesenne Twister implementation provided by Makoto Matsumoto,
<a href=http://www.math.sci.hiroshima-u.ac.jp/~m-mat/MT/MT2002/emt19937ar.html>available here</a>.
You may try using your system's rand() function and srand() functions, but keep in mind that Windows' generator produces numbers in a very small range.

The compacting hash table and multivalue hash tables require using <a href="http://code.google.com/p/cudpp/">CUDPP</a>
and <a href="http://code.google.com/p/back40computing/wiki/RadixSorting">Duane Merrill's radix sort</a>.

\subsection Compiling
There are three options for compiling the library.  The bold ones are default.
<table style="width:90%; margin: auto; border: 1px solid #dddddd;">
  <tr>
    <th class="classes" style="padding-right: 1em;">
      TYPE
    </th>
    <td>
      <b>release</b> or debug
    </td>
    <td>
      "debug" mode analyzes the hash functions by running various tests, and double-checks to see if the query results are correct.  Don't use it for normal use.
    </td>
  </tr>
  <tr>
    <th class="classes" style="padding-right: 1em;">
      ARCH
    </th>
    <td>
      sm_13 or <b>sm_20</b>
    </td>
    <td>
      Architecture to compile for.
    </td>
  </tr>
  <tr>
    <th class="classes" style="padding-right: 1em;">
      BUILD
    </th>
    <td>
      full or <b>basic</b>
    </td>
    <td>
      Compiling the full library adds the compacting and multi-value hash tables.
    </td>
  </tr>
</table>

Typing "make" is equivalent to "make TYPE=release ARCH=sm_20 BUILD=basic".  The required header files and compiled libraries will be placed in the sub-directory <b>../install/</b>.

\section Samples Sample applications
\subsection SimpleSample Simple hash table sample
This program builds a basic hash table using N random key-value pairs with unique keys, then queries it for N unique keys, where the queries are comprised of keys both
inside the hash table and not in the hash table.
Multiple copies of the hash table are built for each trial, where each hash table has a different number of slots.

After the construction of each hash table, it is queried multiple times with a different set of keys.  Each query key set is composed of a portion of the original
input keys (which can be found in the hash table), and keys that were not part of the original input (which cause the queries to fail).

To compile it, type "make simple_sample".
It takes a single command line argument, which says how many items are within the table.

\subsection CompactingSample Compacting hash table example
This builds a compacting hash table, using N random keys.  Multiple copies of a key in the input are all given the same unique ID by the hash table.
In addition to the trials performed by the simple hash table sample, it also performs multiple trials with an increasing average number of copies for each key:
the compacting hash table is always handed N keys, but it is possible that many keys will have a large number of copies.

To compile it, type "make BUILD=full compacting_sample".

\subsection MultivalueSample Multi-value hash table example
This builds a multi-value hash table, using N random key-value pairs.  A key with multiple values is represented by multiple key-value pairs in the input with the same key.
To compile it, type "make BUILD=full multivalue_sample".


<!--
\section Help Getting help and reporting problems
\section Credits
-->

\section License
\include "license.txt"
*/
