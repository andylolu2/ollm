; Taken from https://gist.github.com/stephenroller/85b41a88c611a260530e4053f6ef84b9
; Run this code in a Clojure REPL (https://www.mycompiler.io/new/clojure) to generate the hearst.rules file
; Copyright Facebook 2018-2020
; Licensed under MIT.

(def noun "[{tag:/N.+/}]")
(def simple-noun-phrase (format "%s+" noun noun))
(def noun-phrase (format "(%s /of|--|'s/ )?%s" simple-noun-phrase simple-noun-phrase))
(def premodifier "(([{tag:/RB.?/}] )*([{tag:JJ}] | [{tag:JJR}] | [{tag:JJS}] | [{tag:VBN}]))?")
(def premodifier-adjonly "(([{tag:/RB.?/}] )*([{tag:JJ}] | [{tag:JJR}] | [{tag:JJS}]))?")
(def head-phrase-adjonly  (format "/\"/?(?$prehead %s) (?$head %s)/\"/?" premodifier-adjonly noun-phrase))
(def head-phrase (format "/\"/?(?$prehead %s) (?$head %s)/\"/?" premodifier noun-phrase))
(def tail-phrase (format "/\"/?(?$pretail %s) (?$tail %s)/\"/?" premodifier noun-phrase))

(def pattern-tail-head
  [{:id "example of" :regex "/which/? [{lemma:be}] /an?/ [{tag:JJ}]? [{lemma:/subgenus|example|class|group|form|type|kind/}] /of/"}
   {:id "and other 0" :regex "/and|or/ /any|some/? /other/"}
   {:id "which is called" :regex "/which/ [{lemma:be}] /called/"}
   {:id "is JJS" :regex "[{lemma:be}] [{tag:JJS}] /most/?"}
   ])

(def token-patterns
  (concat
   (map
     (fn [p]
       {:id (p :id)
       :regex (format "(%s/,/? %s %s)" tail-phrase (p :regex) head-phrase)})
     pattern-tail-head)
   [{:id "special case of" :regex (format "(%s [{lemma:be}] /a/ /special/ /case/ /of/ [{tag:DT}]? %s)" tail-phrase head-phrase)}
    {:id "is an X that" :regex (format "(%s [{lemma:be}] /an?/? %s /that/)" tail-phrase head-phrase-adjonly)}
    {:id "is a [!part]" :regex (format "([{tag:DT}] %s [{lemma:be}] /an?/ [!{word:/member|part|given/}] %s)" tail-phrase head-phrase)}
    {:id "!features such as" :regex (format "(([!{word:/features|properties/}] [!{word:/of/}]) %s /such/ /as/ %s)" head-phrase tail-phrase)}
    {:id "such as 1" :regex (format "(%s/,/? /such/ /as/ (?:%s/,/ ) %s)" head-phrase noun-phrase tail-phrase)}
    {:id "such as 2" :regex (format "(%s/,/? /such/ /as/ (?:%s/,/ ){2} %s)" head-phrase noun-phrase tail-phrase)}
    {:id "such as 3" :regex (format "(%s/,/? /such/ /as/ (?:%s/,/ ){3} %s)" head-phrase noun-phrase tail-phrase)}
    {:id "such as 4" :regex (format "(%s/,/? /such/ /as/ (?:%s/,/ ){4} %s)" head-phrase noun-phrase tail-phrase)}
    {:id "such as 5" :regex (format "(%s/,/? /such/ /as/ (?:%s/,/ ){5} %s)" head-phrase noun-phrase tail-phrase)}
    {:id "such as 6" :regex (format "(%s/,/? /such/ /as/ (?:%s/,/ ){6} %s)" head-phrase noun-phrase tail-phrase)}
    {:id "such as &" :regex (format "(%s/,/? /such/ /as/ (?:%s/,/ ){1,9} %s/,/? /and/ %s)" head-phrase noun-phrase noun-phrase tail-phrase)}
    {:id "and other 1" :regex (format "(%s/,/ %s/,/? /and|or/ /any|some/? /other/ %s)" tail-phrase noun-phrase head-phrase)}
    {:id "and other 2" :regex (format "(%s (?:/,/ %s){2}/,/? /and|or/ /any|some/? /other/ %s)" tail-phrase noun-phrase head-phrase)}
    {:id "and other 3" :regex (format "(%s (?:/,/ %s){3}/,/? /and|or/ /any|some/? /other/ %s)" tail-phrase noun-phrase head-phrase)}
    {:id "and other 4" :regex (format "(%s (?:/,/ %s){4}/,/? /and|or/ /any|some/? /other/ %s)" tail-phrase noun-phrase head-phrase)}
    {:id "and other 5" :regex (format "(%s (?:/,/ %s){5}/,/? /and|or/ /any|some/? /other/ %s)" tail-phrase noun-phrase head-phrase)}
    {:id "like most" :regex (format "(/Unlike|Like/ /most|all|any|other/ %s/,/ [{tag:DT}]? %s)" head-phrase tail-phrase)}
    {:id "including 0" :regex (format "(%s/,/? /including/ %s)" head-phrase tail-phrase)}
    {:id "including 1" :regex (format "(%s/,/? /including/ (?: %s/,/ ) %s)" head-phrase noun-phrase tail-phrase)}
    {:id "including 2" :regex (format "(%s/,/? /including/ (?: %s/,/ ){2} %s)" head-phrase noun-phrase tail-phrase)}
    {:id "including 3" :regex (format "(%s/,/? /including/ (?: %s/,/ ){3} %s)" head-phrase noun-phrase tail-phrase)}
    {:id "including 4" :regex (format "(%s/,/? /including/ (?: %s/,/ ){4} %s)" head-phrase noun-phrase tail-phrase)}
    {:id "including 5" :regex (format "(%s/,/? /including/ (?: %s/,/ ){5} %s)" head-phrase noun-phrase tail-phrase)}
    {:id "including &" :regex (format "(%s/,/? /including/ (?: %s)(?:/,/ %s){0,10}/,/? and %s)" head-phrase noun-phrase noun-phrase tail-phrase)}
    ]))

(defn print-patterns [patterns]
  (doseq [pattern patterns]
    (println "{ ruleType: \"tokens\", pattern:" (pattern :regex) ", action: Annotate($head, ner, Concat($$pretail.text, \" \", $$tail.text, \"|||\", $$prehead.text, \" \", $$head.text, \"|||\", \"" (pattern :id) "\")) }"
    )))

(print-patterns token-patterns)