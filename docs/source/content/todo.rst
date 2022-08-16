.. todo::

   * Return blocks in the correct order.
   * Add tests to ensure
    * Read data is correct
    * Reading N bytes past file raises an error
    * Reading non-512-multiple bytes works as expected.
    * Numpy array memory management is working as expected.
   * Add mechanism to deal with multiple files (?? or caller supplies fd ??)
   * Test with multiple files
