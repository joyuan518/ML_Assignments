## Copyright (C) 2016 袁东
## 
## This program is free software; you can redistribute it and/or modify it
## under the terms of the GNU General Public License as published by
## the Free Software Foundation; either version 3 of the License, or
## (at your option) any later version.
## 
## This program is distributed in the hope that it will be useful,
## but WITHOUT ANY WARRANTY; without even the implied warranty of
## MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
## GNU General Public License for more details.
## 
## You should have received a copy of the GNU General Public License
## along with this program.  If not, see <http://www.gnu.org/licenses/>.

## -*- texinfo -*- 
## @deftypefn {Function File} {@var{retval} =} ut_sigmoid (@var{input1}, @var{input2})
##
## @seealso{}
## @end deftypefn

## Author: 袁东 <yuandong@yuandongdeMacBook-Air.local>
## Created: 2016-10-22

function ut_sigmoid()
  % unit test of sigmoid function
  val = 0;
  
  if (sigmoid(val) == 0.5)
    fprintf("when val = 0, result should = 0.5; test passed\n");
  else
    fprintf("when val = 0, result should = 0.5; test failed\n");
  endif
  
  val = 10000;
  
  if (sigmoid(val) == 1)
    fprintf("when val = 10000, result should = 1; test passed\n");
  else
    fprintf("when val = 10000, result should = 1; test failed\n");
  endif
  
  val = -10000;
  
  if (sigmoid(val) == 0)
    fprintf("when val = -10000, result should = 0; test passed\n");
  else
    fprintf("when val = -10000, result should = 0; test failed\n");
  endif
  
  mtx = [10000 0 -10000; 0 -10000 10000];
  
  if (isequal(sigmoid(mtx), [1 0.5 0; 0.5 0 1]))
    fprintf("when mtx = [10000 0 -10000; 0 -10000 10000], result should = [1 0.5 0; 0.5 0 1]; test passed\n");
  else
    fprintf("when mtx = [10000 0 -10000; 0 -10000 10000], result should = [1 0.5 0; 0.5 0 1]; test failed\n");
  endif
  
end  
