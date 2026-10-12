# frozen_string_literal: true

source "https://rubygems.org"

gemspec

gem "jemoji"
gem "jekyll-feed", "~> 0.17"
gem "webrick", "~> 1.8"

gem "html-proofer", "~> 5.0", group: :test

platforms :windows, :jruby do
  gem "tzinfo", ">= 1", "< 3"
  gem "tzinfo-data"
end

gem "wdm", "~> 0.2.0", :platforms => [:windows]
