defmodule StephenColbertTest do
  use ExUnit.Case
  doctest StephenColbert

  test "greets the world" do
    assert StephenColbert.hello() == :world
  end
end
