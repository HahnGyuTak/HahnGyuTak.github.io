# UI adaptations only: keep the original post Markdown intact.
require 'nokogiri'
require 'cgi'
require 'uri'

module PaperReviewUI
  ICON_PATH = '/assets/img/ui'
  ICON_DIRECTORY = File.expand_path("..#{ICON_PATH}", __dir__)

  def self.destination_icon(href, destinations)
    uri = URI.parse(href)
    return {} unless uri.is_a?(URI::HTTP) && uri.host

    host = uri.host.downcase.delete_prefix('www.')
    destinations[host + uri.path.delete_suffix('/')] || destinations.fetch(host, {})
  rescue URI::InvalidURIError
    {}
  end

  def self.icon(kind, favicon, baseurl)
    if favicon
      src = CGI.escapeHTML("#{baseurl}#{ICON_PATH}/#{kind}.png")
      return %(<img src="#{src}" alt="" width="24" height="24">)
    end

    File.read(File.join(ICON_DIRECTORY, "#{kind}.svg"))
        .sub('<svg ', '<svg aria-hidden="true" focusable="false" ')
  end

  def self.decorate(post)
    return unless Array(post.data['categories']).include?('Paper Review')
    document = Nokogiri::HTML(post.output)
    content = document.at_css('article[data-paper-review] > .content')
    return unless content
    destinations = post.site.data.fetch('paper-resource-icons', {})
    baseurl = post.site.config['baseurl'].to_s.delete_suffix('/')
    icon_credits = []
    content.css('details').each do |details|
      summary = details.at_xpath('./summary')
      next unless summary
      details.add_class('review-detail')
      body = Nokogiri::XML::Node.new('div', document)
      body['class'] = 'review-detail-content'
      details.children.to_a.each { |child| body.add_child(child) unless child == summary }
      details.add_child(body)
    end
    # Only the introductory bibliography, before Background/other H2 sections.
    content.element_children.take_while { |node| node.name != 'h2' }.each do |node|
      next unless node.name == 'blockquote'
      node.css('a[href]').each do |link|
        label = link.text.strip
        fallback = case label
                   when /\A(?:arxiv|paper|pdf|논문)(?:\s.*)?\z/i then 'paper'
                   when /\A(?:github|code|코드)(?:\s.*)?\z/i then 'code'
                   when /\A(?:project|프로젝트|hugging\s*face)(?:\s.*)?\z/i then 'project'
                   end
        next unless fallback

        destination = destination_icon(link['href'], destinations)
        kind = destination.fetch('kind', fallback)
        icon_credits << destination['credit'] if destination['credit']
        link.previous_sibling.content = link.previous_sibling.content.sub(/\[\s*\z/, '') if link.previous_sibling&.text?
        link.next_sibling.content = link.next_sibling.content.sub(/\A\s*\]/, '') if link.next_sibling&.text?
        link.add_class('paper-resource-button')
        link['data-resource'] = kind
        link['aria-label'] = "#{label} — 새 탭에서 열기"
        link['target'] = '_blank'
        link['rel'] = [link['rel'], 'noopener', 'noreferrer'].compact.join(' ').split.uniq.join(' ')
        link.inner_html = '<span class="resource-decor" aria-hidden="true"></span>' \
          '<span class="resource-content"><span class="resource-icon" aria-hidden="true">' + \
          icon(kind, destination['favicon'], baseurl) + \
          '</span><span class="resource-label">' + CGI.escapeHTML(label) + '</span></span>'
        link.parent.add_class('paper-resources')
      end
    end
    footer = document.at_css('footer')
    icon_credits.uniq.each do |credit|
      footer&.add_child('<p><a href="' + CGI.escapeHTML(credit.fetch('url')) + '">' +
        CGI.escapeHTML(credit.fetch('text')) + '</a></p>')
    end
    post.output = document.to_html
  end
end

Jekyll::Hooks.register :posts, :post_render do |post|
  PaperReviewUI.decorate(post)
end
