# frozen_string_literal: true

# Tags differing only in case share a URL. Combine their archive entries while
# leaving the tags in the original post front matter unchanged.
Jekyll::Hooks.register :site, :post_read do |site|
  tags = site.tags
  groups = tags.keys.group_by { |name| Jekyll::Utils.slugify(name) }

  groups.each_value do |names|
    next if names.length < 2

    posts = names.flat_map { |name| tags.fetch(name) }.uniq.sort_by(&:date).reverse
    canonical = names.first
    names.each { |name| tags.delete(name) }
    tags[canonical] = posts
  end
end
